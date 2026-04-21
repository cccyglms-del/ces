# Deployment Guide

This project is intended to run as a hosted Streamlit app.

## Required Files

Commit these files and folders:

- `app.py`
- `kmtool/`
- `sample_data/`
- `tools/`
- `tests/`
- `requirements.txt`
- `runtime.txt`
- `packages.txt`
- `.streamlit/config.toml`

Do not commit:

- `.venv/`
- `__pycache__/`
- `.streamlit/secrets.toml`
- `data/`

## Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to `app.py`.
5. Deploy the app.

Optional secrets:

```toml
LLM_API_KEY = "your_key"
LLM_MODEL = "gpt-4.1-mini"
LLM_API_BASE = "https://api.openai.com/v1"
```

## Local macOS Run

```bash
brew install python@3.11 tesseract
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
streamlit run app.py
```

## Cache Behavior

The app uses Streamlit's in-memory cache for:

- PDF page rendering
- PubMed / Europe PMC literature search results

The cache is not a database. It can be cleared when the app restarts or the hosting platform redeploys.
