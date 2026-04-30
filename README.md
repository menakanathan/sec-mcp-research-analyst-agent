# SEC MCP Research Analyst Agent with Local LLM

This version uses a local LLM through Ollama instead of OpenAI API.

## Setup

```bash

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```text
SEC_USER_AGENT="Your Name your.email@example.com"
LLM_PROVIDER="ollama"
OLLAMA_MODEL="llama3.1:8b"
OLLAMA_BASE_URL="http://localhost:11434"
```

## Install and run Ollama

Install Ollama from https://ollama.com, then run:

```bash
ollama pull llama3.1:8b
ollama serve
```

If your machine is limited, use:

```bash
ollama pull llama3.2:3b
```

and set:

```text
OLLAMA_MODEL="llama3.2:3b"
```

## Test

```bash
python -m app.llm_cli "Create an analyst brief for Apple using latest 10-K"
python -m app.llm_cli "Create an analyst brief for Apple using latest 10-K" --no-llm
```

## API

```bash
uvicorn app.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Useful endpoints:

```text
GET  /health
GET  /llm-health
POST /ask
GET  /brief/{ticker}
GET  /facts/{ticker}
GET  /section/{ticker}
```

## MCP Server

```bash
python -m app.mcp_server
```

This project is for academic research and is not financial advice.


## Streamlit UI

Run the simple UI:

```bash
streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

The UI includes:

- question input
- sample questions
- local LLM health check
- analyst response
- tools used
- structured evidence pack
- raw JSON view
