import gc
import os
import re
import shutil
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

import fitz
import numpy as np
import openai
import torch
import torch.nn.functional as F
from fastapi import UploadFile
from lcserve import serving
from optimum.bettertransformer import BetterTransformer
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

recommender = None


def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = re.sub("\s+", " ", text)
    return text


def get_margin(pdf):
    page = pdf[0]
    page_size = page.mediabox
    margin_hor = page.mediabox.width * 0.05
    margin_ver = page.mediabox.height * 0.05
    margin_size = page_size + (margin_hor, margin_ver, -margin_hor, -margin_ver)
    return margin_size


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []
    margin_size = get_margin(doc)
    for i in range(start_page - 1, end_page):
        page = doc[i]
        page.set_cropbox(margin_size)
        text = page.get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(" ") for t in texts]
    page_nums = []
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i : i + word_length]
            if (
                (i + word_length) > len(words)
                and (len(chunk) < word_length)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = " ".join(chunk).strip()
            chunk = f"[Page no. {idx+start_page}]" + " " + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    def __init__(self, embedding_model):
        self.tokenizer = AutoTokenizer.from_pretrained(f"intfloat/{embedding_model}")
        self.model = AutoModel.from_pretrained(
            f"intfloat/{embedding_model}",
            # cache_dir =,
        )
        self.model = BetterTransformer.transform(self.model, keep_original_model=True)

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.fitted = False

    def fit(self, data, batch_size=32, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(self.data, batch_size=batch_size)
        self.fitted = True

    def __call__(self, text, return_data=True):
        self.inp_emb = self.get_text_embedding([text], prefix="query")
        self.matches = self.run_svm(self.inp_emb, self.embeddings)

        if return_data:
            # return 5 first match, first index is query, so it has to be skipped
            return [self.data[i - 1] for i in self.matches[1:6]]

        else:
            return self.matches

    def average_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        self.last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return self.last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_text_embedding(self, texts, prefix="passage", batch_size=32):
        # Tokenize the input texts
        texts = [f"{prefix}: {text}" for text in texts]
        batch_dict = self.tokenizer(
            texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)

        embeddings = self.average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Convert pytorch tensor to numpy array (no grad)
        if self.device == "cuda":
            embeddings = embeddings.detach().cpu().clone().numpy()
        else:
            embeddings = embeddings.detach().numpy()
        return embeddings

    def run_svm(self, query_emb, passage_emb):
        joined_emb = np.concatenate((query_emb, passage_emb))

        # create var for SVM label
        y = np.zeros(joined_emb.shape[0])
        # mark query as a positive example
        y[0] = 1

        # declare SVM
        clf = svm.LinearSVC(
            class_weight="balanced", verbose=False, max_iter=10000, tol=1e-6, C=0.1
        )
        # train (Exemplar) SVM
        clf.fit(joined_emb, y)

        # infer on original data
        similarities = clf.decision_function(joined_emb)
        sorted_ix = np.argsort(-similarities)
        return sorted_ix

    def summarize(self):
        n_clusters = int(np.ceil(len(self.embeddings)**0.5))
        # max cluster 5 (reserve token)
        n_clusters = n_clusters if n_clusters <= 5 else 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=23)
        kmeans = kmeans.fit(self.embeddings)

        avg = []
        closest = []
        for j in range(n_clusters):
            # find first chunk index of every cluster
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        # find chunk that is closest to the centroid
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_,
                                                   self.embeddings)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        # concat representative chunks
        summary = [self.data[i] for i in [closest[idx] for idx in ordering]]
        return summary


def clear_cache():
    global recommender
    if "recommender" in globals():
        del recommender
    gc.collect()
    if torch.cuda.is_available():
        return torch.cuda.empty_cache()


def load_recommender(path, embedding_model, rebuild_embedding, start_page=1):
    global recommender
    if rebuild_embedding:
        clear_cache()
        recommender = None
    if recommender is None:
        recommender = SemanticSearch(embedding_model)
    if recommender.fitted:
        return "Corpus Loaded."
    else:
        texts = pdf_to_text(path, start_page=start_page)
        chunks = text_to_chunks(texts, start_page=start_page)
        recommender.fit(chunks)
    return "Corpus Loaded."


def generate_text(openai_key, prompt, model="gpt-3.5-turbo"):
    openai.api_key = openai_key
    completions = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = f"{prompt}###{completions.choices[0].message.content}###{completions.usage.total_tokens}###{completions.model}"
    return message

def generate_answer(question, gpt_model, openai_key):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += "search results:\n\n"
    for c in topn_chunks:
        prompt += c + "\n\n"

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "
        "Citation should be done at the end of each sentence. If the search results mention multiple subjects "
        "with the same name, create separate answers for each. Only include information found in the results and "
        "don't add any additional information. Make sure the answer is correct and don't output false content. "
        "If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier "
        "search results which has nothing to do with the question. Only answer what is asked. The "
        "answer should be short and concise. Answer step-by-step.\n\n"
    )

    prompt += f"Query: {question}"
    answer = generate_text(openai_key, prompt, gpt_model)
    return answer

def generate_summary(gpt_model, openai_key):
    topn_chunks = recommender.summarize()
    prompt = ""
    prompt += (
        "Summarize the highlights of the search results and output a summary in bulletpoints. "
        "Do not write anything before the bulletpoints. "
        "Cite each reference using [Page no.] notation (every result has this number at the beginning). "
        "Citation should be done at the end of each sentence. "
        "Give conclusion in the end. "
        "Write summary in the same language as the search results. "
        "Search results:\n\n"
    )
    for c in topn_chunks:
        prompt += c + "\n\n"
    summary = generate_text(openai_key, prompt, gpt_model)
    return summary


def load_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key is None:
        raise ValueError(
            "[ERROR]: Please pass your OPENAI_API_KEY. Get your key here : https://platform.openai.com/account/api-keys"
        )
    return key


# %%
@serving
def ask_url(
    url: str,
    question: str,
    rebuild_embedding: bool,
    embedding_model: str,
    gpt_model: str,
) -> str:
    if rebuild_embedding:
        load_url(url, embedding_model, rebuild_embedding)
    openai_key = load_openai_key()
    return generate_answer(question, gpt_model, openai_key)


@serving
async def ask_file(
    file: UploadFile,
    question: str,
    rebuild_embedding: bool,
    embedding_model: str,
    gpt_model: str,
) -> str:
    if rebuild_embedding:
        load_file(file, embedding_model, rebuild_embedding)
    openai_key = load_openai_key()
    return generate_answer(question, gpt_model, openai_key)


@serving
def load_url(url: str,
             embedding_model: str,
             rebuild_embedding: bool,
             gpt_model: str
             ) -> str:
    download_pdf(url, "corpus.pdf")
    notification = load_recommender("corpus.pdf", embedding_model, rebuild_embedding)
    openai_key = load_openai_key()
    summary = generate_summary(gpt_model, openai_key)
    response = f"{notification}###{summary}"
    return response


@serving
async def load_file(
    file: UploadFile,
        embedding_model: str,
        rebuild_embedding: bool,
        gpt_model: str
) -> str:
    suffix = Path(file.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    notification = load_recommender(str(tmp_path), embedding_model, rebuild_embedding)
    openai_key = load_openai_key()
    summary = generate_summary(gpt_model, openai_key)
    response = f"{notification}###{summary}"
    return response
