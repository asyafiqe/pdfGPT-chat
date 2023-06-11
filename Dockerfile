FROM python:3.9-slim-bullseye as pdfgpt-chat

COPY requirements_pytorch.txt requirements_pytorch.txt
COPY requirements_other.txt requirements_other.txt
RUN pip3 install -r requirements_pytorch.txt
RUN pip3 install -r requirements_other.txt

WORKDIR /app

COPY intfloat /app/intfloat
COPY app.py app.py
COPY api.py api.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT  ["streamlit", "run", "app.py", "--server.port=8501"]