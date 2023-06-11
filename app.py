# %%
import json
import urllib.parse
from tempfile import _TemporaryFileWrapper

import pandas as pd
import requests
import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

st.set_page_config(
    layout="wide",
    page_title="pdfGPT-chat. Ask your PDF!",
    page_icon=":robot_face:",
)


def main():
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    def pdf_change():
        st.session_state["pdf_change"] = True

    def check_api(api_key):
        return api_key.startswith("sk-") and len(api_key) == 51

    def check_url(url):
        parsed_url = urllib.parse.urlparse(url)
        return all([parsed_url.scheme, parsed_url.netloc])

    def load_pdf():
        if not check_url(lcserve_host):
            return st.error("Please enter valid API host.")
        elif not check_api(openai_key):
            return st.error("Please enter valid OpenAI API.")
        elif file is None and len(pdf_url) == 0:
            return st.error("Both URL and PDF is empty. Provide at least one.")
        elif len(pdf_url) > 0:
            if not check_url(pdf_url):
                return st.error("Please enter valid URL.")
            elif file is not None:
                return st.error(
                    "Both URL and PDF is provided. Please provide only one (either URL or PDF)."
                )
            # load pdf from url
            else:
                r = requests.post(
                    f"{lcserve_host}/load_url",
                    json={
                        "url": pdf_url,
                        "rebuild_embedding": st.session_state["pdf_change"],
                        "embedding_model": embedding_model,
                    },
                )
        # load file
        else:
            _data = {
                "rebuild_embedding": st.session_state["pdf_change"],
                "embedding_model": embedding_model,
            }

            r = requests.post(
                f"{lcserve_host}/load_file",
                params={"input_data": json.dumps(_data)},
                files={"file": file},
            )
        if r.status_code != 200:
            if "error" in r.json():
                if "message" in r.json()["error"]:
                    return st.error(r.json()["error"]["message"])
            else:
                return str(r.json())
        elif r.json()["result"] == "Corpus Loaded.":
            st.session_state["loaded"] = True
            st.session_state["pdf_change"] = False
            return st.success("The PDF file has been loaded.")
        else:
            return st.info(r.json()["result"])

    def generate_response(
        lcserve_host: str,
        url: str,
        file: _TemporaryFileWrapper,
        question: str,
        openai_key: str,
    ) -> dict:
        if question.strip() == "":
            return "[ERROR]: Question field is empty"

        _data = {
            "question": question,
            "rebuild_embedding": st.session_state["pdf_change"],
            "embedding_model": embedding_model,
            "gpt_model": gpt_model,
            "envs": {
                "OPENAI_API_KEY": openai_key,
            },
        }

        if url.strip() != "":
            r = requests.post(
                f"{lcserve_host}/ask_url",
                json={"url": url, **_data},
            )

        else:
            r = requests.post(
                f"{lcserve_host}/ask_file",
                params={"input_data": json.dumps(_data)},
                files={"file": file},
            )

        if r.status_code != 200:
            content = r.content.decode()  # Convert bytes to string
            with open("langchainlog.txt", "w") as file:
                file.write(content)
            return f"[ERROR]: {r.text}"

        result = r.json()["result"]
        result = result.split("###")
        keys = ["prompt", "answer", "token_used", "gpt_model"]
        # Error in OpenAI server also gives status_code 200
        if len(result) >= 0:
            result.extend([result, 0, gpt_model])

        result_dict = dict(zip(keys, result))
        return result_dict

    # %%
    # main page layout
    header = st.container()
    welcome_page = st.container()
    response_container = st.container()
    input_container = st.container()
    cost_container = st.container()
    load_pdf_popup = st.container()

    # sidebar layout
    input_details = st.sidebar.container()
    preferences = st.sidebar.container()
    chat_download = st.sidebar.container()
    # %%
    # instantiate session states
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = False

    if "generated" not in st.session_state:
        st.session_state["generated"] = [
            "Hello there. I'm pdfGPT-chat. Do you have any question about your PDF?"
        ]

    if "loaded" not in st.session_state:
        st.session_state["loaded"] = False

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hi"]

    if "pdf_change" not in st.session_state:
        st.session_state["pdf_change"] = True

    if "total_cost" not in st.session_state:
        st.session_state["total_cost"] = 0

    if "total_token" not in st.session_state:
        st.session_state["total_token"] = 0

    # %%
    # constants
    E5_URL = "https://github.com/microsoft/unilm/tree/master/e5"
    EMBEDDING_CHOICES = {
        "multilingual-e5-base": "Multilingual-E5 (default)",
        "e5-small-v2": "English-E5-small (faster)",
    }
    GPT_CHOICES = {
        "gpt-3.5-turbo": "GPT-3.5-turbo (default)",
        "gpt-4": "GPT-4 (smarter, costlier)",
    }
    PDFGPT_URL = "https://github.com/bhaskatripathi/pdfGPT"
    SIGNATURE = """<style>
.footer {
position: static;
left: 0;
bottom: 0;
width: 100%;
background: rgba(0,0,0,0);
text-align: center;
}
</style>

<div class="footer">
<p style='display: block;
text-align: center;
font-size:14px;
color:darkgray'>Developed with ❤ by asyafiqe</p>
</div>
"""

    with header:
        st.title(":page_facing_up: pdfGPT-chat")
        with st.expander(
            "A fork of [pdfGPT](%s) with several improvements. With pdfGPT-chat, you can chat with your PDF files using [**Microsoft E5 Multilingual Text Embeddings**](%s) and **OpenAI**."
            % (PDFGPT_URL, E5_URL)
        ):
            st.markdown(
                "Compared to other tools, pdfGPT-chat provides **hallucinations-free** response, thanks to its superior embeddings and tailored prompt.<br />The generated responses from pdfGPT-chat include **citations** in square brackets ([]), indicating the **page numbers** where the relevant information is found.<br />This feature not only enhances the credibility of the responses but also aids in swiftly locating the pertinent information within the PDF file.",
                unsafe_allow_html=True,
            )

        colored_header(
            label="",
            description="",
            color_name="blue-40",
        )

    with preferences:
        colored_header(
            label="",
            description="",
            color_name="blue-40",
        )
        st.write("**Preferences**")
        embedding_model = st.selectbox(
            "Embedding",
            EMBEDDING_CHOICES.keys(),
            help="""[Multilingual-E5](%s) supports 100 languages. 
            E5-small is much faster and suitable for PC without GPU."""
            % E5_URL,
            on_change=pdf_change,
            format_func=lambda x: EMBEDDING_CHOICES[x],
        )
        gpt_model = st.selectbox(
            "GPT Model",
            GPT_CHOICES.keys(),
            help="For GPT-4 you might have to join the waitlist: https://openai.com/waitlist/gpt-4-api",
            format_func=lambda x: GPT_CHOICES[x],
        )

    # %%
    # sidebar
    with input_details:
        # sidebar
        st.title("Input details")
        lcserve_host = st.text_input(
            label=":computer: Enter your API Host here",
            value="http://localhost:8080",
            placeholder="http://localhost:8080",
            autocomplete="http://localhost:8080",
            help="Your langchain-serve host, default is http://localhost:8080",
        )

        OPENAI_URL = "https://platform.openai.com/account/api-keys"
        openai_key = st.text_input(
            ":key: Enter your OpenAI API key here",
            type="password",
            help="Get your Open AI API key [here](%s)" % OPENAI_URL,
        )
        colored_header(
            label="",
            description="",
            color_name="blue-40",
        )

        pdf_url = st.text_input(
            ":globe_with_meridians: Enter PDF URL here", on_change=pdf_change
        )

        st.markdown(
            "<h2 style='text-align: center; color: black;'>OR</h2>",
            unsafe_allow_html=True,
        )

        file = st.file_uploader(
            ":page_facing_up: Upload your PDF/ Research Paper / Book here",
            type=["pdf"],
            on_change=pdf_change,
        )

        if st.button("Load PDF"):
            st.session_state["loaded"] = True
            with st.spinner("Loading PDF"):
                with load_pdf_popup:
                    load_pdf()

    # %%

    # main tab
    if st.session_state["loaded"]:
        with input_container:
            with st.form(key="input_form", clear_on_submit=True):
                user_input = st.text_area("Question:", key="input", height=100)
                submit_button = st.form_submit_button(label="Send")

            if user_input and submit_button:
                with st.spinner("Processing your question"):
                    response = generate_response(
                        lcserve_host,
                        pdf_url,
                        file,
                        user_input,
                        openai_key,
                    )
                    st.session_state.past.append(user_input)
                    st.session_state.generated.append(response["answer"])

                    # calculate cost
                    st.session_state["total_token"] += int(response["token_used"])
                    if "gpt-3" in response["gpt_model"]:
                        current_cost = st.session_state["total_token"] * 0.002 / 1000
                    else:
                        current_cost = st.session_state["total_token"] * 0.06 / 1000
                    st.session_state["total_cost"] += current_cost

            if not user_input and submit_button:
                st.error("Please write your question.")

        with response_container:
            if st.session_state["generated"]:
                for i in range(len(st.session_state["generated"])):
                    message(
                        st.session_state["past"][i], is_user=True, key=str(i) + "_user"
                    )
                    message(st.session_state["generated"][i], key=str(i))

        cost_container.caption(
            f"Estimated cost: $ {st.session_state['total_cost']:.4f}"
        )

    else:
        with welcome_page:
            st.write("")
            st.subheader(
                """:arrow_left: To start please fill input details in the sidebar and click **Load PDF**"""
            )
    # %%
    # placed in the end to include the last conversation
    with chat_download:
        chat_history = pd.DataFrame(
            {
                "Question": st.session_state["past"],
                "Answer": st.session_state["generated"],
            }
        )

        csv = convert_df(chat_history)

        st.download_button(
            label="Download chat history",
            data=csv,
            file_name="chat history.csv",
            mime="text/csv",
        )
        add_vertical_space(2)
        st.markdown(SIGNATURE, unsafe_allow_html=True)

    # %%
    # javascript

    # scroll halfway through the page
    js = f"""
    <script>
    function scroll() {{
    var textAreas = parent.document.querySelectorAll('section.main');
    var halfwayScroll = 0.5 * textAreas[0].scrollHeight; // Calculate halfway scroll position
    
    for (let index = 0; index < textAreas.length; index++) {{
    textAreas[index].scrollTop = halfwayScroll; // Set scroll position to halfway
    }}
    }}

    scroll(); // Call the scroll function
    </script>
    """
    st.components.v1.html(js)

    # reduce main top padding
    st.markdown(
        "<style>div.block-container{padding-top:1.5em;}</style>",
        unsafe_allow_html=True,
    )
    # reduce sidebar top padding
    st.markdown(
        "<style>.css-ysnqb2.e1g8pov64 {margin-top: -90px;}</style>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
