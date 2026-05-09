# SHL Conversational Assessment Recommender — Approach Document

## Design Overview

### Architecture

A stateless FastAPI service with two layers:

1. **Retrieval layer** — TF-IDF index over the full SHL catalog (~400+ items). On each `/chat` call, the last 3 user turns are used as a retrieval query. The top 25 most relevant catalog items are injected into the system prompt as structured context. Items explicitly mentioned by name in the conversation are always included regardless of TF-IDF score.

2. **LLM layer** — An OpenAI-compatible chat completion call (works with GPT-4o-mini, Groq Llama-3.3-70b, or OpenRouter Gemini). The system prompt contains the catalog context, strict behavioral rules, and a JSON output schema. Temperature is set to 0.1 for consistency.

### Why TF-IDF over embeddings?

- Zero cold-start latency: no embedding model to load or API to call.
- The catalog is small enough (~400 items) that TF-IDF recall is competitive.
- No additional API cost or dependency.
- Tradeoff: semantic similarity (e.g. "cognitive test" → "Verify G+") is weaker than with embeddings. Mitigated by injecting a large top-k (25 items) and relying on the LLM to select from the context.

### Prompt Design

The system prompt does four things:
1. **Scopes** the agent to SHL assessments only, with explicit refusal rules.
2. **Injects** the retrieved catalog context as structured text (name, URL, type, duration, languages, description).
3. **Enforces** the exact JSON output schema — no markdown fences, no deviation.
4. **Adds turn-awareness** at turn 6+ to force commitment before the 8-turn cap.

The LLM is instructed to return raw JSON always. The parser strips markdown fences defensively and validates every recommended URL against the catalog. Hallucinated URLs are dropped silently; hallucinated names are resolved via fuzzy name lookup.

### Agent Behaviors

| Behavior | Implementation |
|---|---|
| Clarify vague queries | Prompt rule: "ask ONE clarifying question before recommending" |
| Recommend 1–10 | Prompt rule + schema enforcement |
| Refine mid-conversation | Full history passed each turn; LLM updates shortlist |
| Compare assessments | Mentioned items always included in context; LLM answers from catalog data |
| Refuse off-topic | Regex check for injection phrases before LLM call; prompt rules for legal/compliance |
| Honor 8-turn cap | Turn counter injected into prompt at turn 6+ |

### Hallucination Prevention

- Every URL in the response is validated against the catalog. Invalid URLs are dropped.
- Names are resolved via exact then fuzzy match against the catalog.
- The catalog context is injected verbatim — the LLM cannot invent products not in the context window.
- Temperature 0.1 reduces creative drift.

## What Didn't Work

- **Strict JSON mode with GPT-3.5**: Frequently produced malformed JSON or ignored the schema. Switched to GPT-4o-mini / Llama-3.3-70b which reliably follow the schema.
- **Embedding-based retrieval**: Added latency and cost without meaningfully improving recall for this catalog size. TF-IDF with top-k=25 covers the relevant items in practice.
- **Single-turn prompting**: Early versions tried to answer in one shot. The sample conversations showed the agent needs to ask at least one clarifying question for vague queries — this required explicit prompt rules and testing against the "vague query" probe.

## Evaluation Approach

Tested against all 10 public conversation traces manually, checking:
- Schema compliance on every turn
- No hallucinated URLs
- Clarification on vague queries (C1, C9)
- Refinement behavior (C4, C8, C9)
- Comparison answers grounded in catalog (C5, C6)
- Refusal of legal questions (C7)
- End-of-conversation detection

Automated test suite (`test_agent.py`) covers schema, URL validity, vague-query behavior, refinement, comparison, end-of-conversation, and legal refusal.

## Stack

- **FastAPI + Uvicorn** — lightweight, async, standard for ML services
- **OpenAI SDK** — compatible with Groq and OpenRouter for free-tier LLMs; no LangChain overhead
- **Custom TF-IDF** — pure Python, no sklearn dependency; keeps cold-start fast and the retrieval logic fully transparent
- **Groq (free tier)** — llama-3.3-70b-versatile; fast inference, no cost, OpenAI-compatible API
- **Render (free tier)** — simple GitHub-connected deployment, sufficient for the evaluation window

## AI Tools Used

Kiro (AI IDE) was used for code scaffolding and iteration. All design decisions, prompt engineering, and evaluation logic were authored and reviewed manually.
