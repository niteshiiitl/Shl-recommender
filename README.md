# SHL Conversational Assessment Recommender

A FastAPI service that helps hiring managers select SHL assessments through natural conversation.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your API key

# 3. Run locally
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API

### GET /health
```json
{"status": "ok"}
```

### POST /chat
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer, mid-level"}
  ]
}
```

Response:
```json
{
  "reply": "...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Deployment (Render)

1. Push to GitHub
2. Create a new Web Service on [Render](https://render.com)
3. Set environment variables: `OPENAI_API_KEY`, `LLM_MODEL`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## LLM Options

| Provider | Model | Notes |
|---|---|---|
| OpenAI | `gpt-4o-mini` | Best quality, ~$0.001/call |
| Groq | `llama-3.3-70b-versatile` | Free tier, fast |
| OpenRouter | `google/gemini-flash-1.5` | Free tier |

For Groq/OpenRouter, set `OPENAI_BASE_URL` in `.env`.

## Running Tests

```bash
python test_agent.py
```
