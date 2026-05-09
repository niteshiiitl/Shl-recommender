"""
agent logic: retrieval + LLM call + response parsing.
"""
import json
import os
import re
import logging
from typing import List, Dict, Any, Tuple

from openai import OpenAI
from catalog_loader import get_index, get_catalog
from prompts import SYSTEM_PROMPT, build_catalog_context, is_off_topic

logger = logging.getLogger(__name__)


# LLM client (OpenAI-compatible — works with Groq, OpenRouter, etc.)

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")  # e.g. https://api.groq.com/openai/v1
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


# Query extraction helpers

def _extract_query_from_history(messages: List[Dict[str, str]]) -> str:
    """Build a retrieval query from the full conversation history."""
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_texts[-3:])  # last 3 user turns for recency


def _extract_names_mentioned(messages: List[Dict[str, str]]) -> List[str]:
    """Extract assessment names mentioned in the conversation for comparison queries."""
    index = get_index()
    all_names = [item["name"].lower() for item in index.get_all()]
    mentioned = []
    full_text = " ".join(m["content"] for m in messages).lower()
    for item in index.get_all():
        if item["name"].lower() in full_text:
            mentioned.append(item["name"])
    return mentioned


# Retrieval

def retrieve_catalog_context(messages: List[Dict[str, str]], top_k: int = 20) -> Tuple[str, List[Dict]]:
    """
    Retrieve relevant catalog items and return formatted context + raw items.
    Uses TF-IDF search over the full conversation query.
    """
    index = get_index()
    query = _extract_query_from_history(messages)
    
    # Also boost items explicitly mentioned by name
    mentioned_names = _extract_names_mentioned(messages)
    
    results = index.search(query, top_k=top_k)
    
    # Ensure mentioned items are always included
    result_names = {r["name"] for r in results}
    for name in mentioned_names:
        if name not in result_names:
            item = index.get_by_name(name)
            if item:
                results.append(item)
    
    # Cap at top_k + mentioned
    results = results[:top_k + len(mentioned_names)]
    
    context = build_catalog_context(results)
    return context, results


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str, catalog_items: List[Dict]) -> Dict[str, Any]:
    """
    Parse the LLM JSON response. Falls back gracefully on malformed output.
    Validates that all recommended URLs exist in the catalog.
    """
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON: %s", raw[:200])
                return {
                    "reply": raw,
                    "recommendations": [],
                    "end_of_conversation": False,
                }
        else:
            return {
                "reply": raw,
                "recommendations": [],
                "end_of_conversation": False,
            }

    # Validate and sanitize recommendations
    valid_urls = {item["url"] for item in get_catalog()}
    url_to_item = {item["url"]: item for item in get_catalog()}
    name_index = get_index()

    raw_recs = data.get("recommendations") or []
    clean_recs = []
    
    for rec in raw_recs:
        url = rec.get("url", "")
        name = rec.get("name", "")
        
        # URL must be in catalog
        if url in valid_urls:
            catalog_item = url_to_item[url]
            clean_recs.append({
                "name": catalog_item["name"],  # use canonical name
                "url": url,
                "test_type": catalog_item["test_type"],
            })
        else:
            # Try to find by name
            found = name_index.get_by_name(name)
            if found:
                clean_recs.append({
                    "name": found["name"],
                    "url": found["url"],
                    "test_type": found["test_type"],
                })
            else:
                logger.warning("Dropping hallucinated recommendation: %s / %s", name, url)

    # Deduplicate
    seen_urls = set()
    deduped = []
    for r in clean_recs:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped.append(r)

    return {
        "reply": str(data.get("reply", "")),
        "recommendations": deduped[:10],
        "end_of_conversation": bool(data.get("end_of_conversation", False)),
    }


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

REFUSAL_RESPONSE = {
    "reply": "I can only help with SHL assessment selection. I'm not able to advise on legal, compliance, or other topics outside the SHL catalog. What role or use case are you assessing for?",
    "recommendations": [],
    "end_of_conversation": False,
}


def run_agent(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Main agent function. Takes stateless conversation history, returns response dict.
    """
    # Check last user message for prompt injection
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    if is_off_topic(last_user):
        return REFUSAL_RESPONSE

    # Retrieve relevant catalog context (Reduced to 10 to prevent context loss)
    catalog_context, catalog_items = retrieve_catalog_context(messages, top_k=10)

    # Build system prompt with injected catalog
    system = SYSTEM_PROMPT.format(catalog_context=catalog_context)

    # Determine turn number for turn-cap awareness
    turn_number = sum(1 for m in messages if m["role"] == "user")
    
    # Add turn awareness to system prompt if near cap
    if turn_number >= 6:
        system += f"\n\n## Turn awareness\nThis is turn {turn_number} of a maximum 8. You MUST commit to a final shortlist now even if context is incomplete. Set end_of_conversation to true if the user seems satisfied."

    client = _get_client()
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=0.1,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return {
            "reply": "I'm having trouble connecting to the language model. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    return _parse_llm_response(raw, catalog_items)