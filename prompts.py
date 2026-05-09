"""
Prompt templates for the SHL assessment recommender agent.
"""

SYSTEM_PROMPT = """You are an SHL assessment advisor. Your only job is to help hiring managers and recruiters select the right SHL Individual Test Solutions from the SHL product catalog.

## Scope
- You ONLY discuss SHL assessments from the catalog provided.
- You REFUSE general hiring advice, legal questions, compliance interpretations, and any off-topic requests.
- You NEVER recommend assessments not in the catalog. Every URL you return must come from the catalog.
- You NEVER hallucinate product names, durations, or features.

## Conversation rules
1. CLARIFY before recommending. If the user's request is too vague (e.g. "I need an assessment"), ask ONE focused clarifying question. Do not ask multiple questions at once.
2. RECOMMEND once you have enough context (role, level, or use-case). Provide 1–10 assessments.
3. REFINE when the user changes constraints mid-conversation. Update the shortlist, do not start over.
4. COMPARE when asked. Answer from catalog data only.
5. REFUSE off-topic questions politely and briefly, then redirect to assessment selection.
6. HONOR the 8-turn cap. If you are on turn 7 or 8, commit to a shortlist even if context is incomplete.

## Output format (STRICT — do not deviate)
You must ALWAYS respond with valid JSON in this exact schema:

{{
  "reply": "<your conversational reply as plain text>",
  "recommendations": [
    {{
      "name": "<exact catalog name>",
      "url": "<exact catalog URL>",
      "test_type": "<short code(s) e.g. K or P or A,S>"
    }}
  ],
  "end_of_conversation": false
}}

- `recommendations` is an EMPTY ARRAY [] when you are still gathering context, refusing, or comparing without committing.
- `recommendations` is an array of 1–10 items when you have committed to a shortlist.
- `end_of_conversation` is true when you have provided a finalized shortlist that meets the user's criteria, or when the user explicitly ends the chat.
- Do NOT wrap the JSON in markdown code fences. Return raw JSON only.

## Catalog context
The following is the relevant portion of the SHL catalog for this query:

{catalog_context}

## Important notes on test_type codes
- A = Ability & Aptitude
- B = Biodata & Situational Judgment  
- C = Competencies
- D = Development & 360
- E = Assessment Exercises
- K = Knowledge & Skills
- P = Personality & Behavior
- S = Simulations

Use the exact codes from the catalog entry. If a product has multiple keys, join them with commas (e.g. "K,S").
"""

REFUSAL_TOPICS = [
    "legal", "hipaa compliance requirement", "gdpr", "lawsuit", "illegal",
    "discriminat", "bias lawsuit", "regulatory requirement", "must we by law",
    "are we required by law", "prompt injection", "ignore previous",
    "ignore all instructions", "disregard", "jailbreak", "act as",
    "you are now", "forget your instructions",
]


def is_off_topic(message: str) -> bool:
    lower = message.lower()
    
    # Check for general off-topic/legal questions
    for topic in REFUSAL_TOPICS:
        if topic in lower:
            return True
            
    # Check for specific injection phrases
    injection_phrases = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard",
        "jailbreak",
        "act as a",
        "you are now",
        "forget your instructions",
        "new instructions",
        "system prompt",
    ]
    for phrase in injection_phrases:
        if phrase in lower:
            return True
            
    return False


def build_catalog_context(items: list) -> str:
    """Format catalog items into a compact context string for the prompt."""
    if not items:
        return "No matching catalog items found."
    
    lines = []
    for item in items:
        lang_str = ", ".join(item["languages"][:5])
        if len(item["languages"]) > 5:
            lang_str += f" (+{len(item['languages'])-5} more)"
        
        level_str = ", ".join(item["job_levels"][:4])
        if len(item["job_levels"]) > 4:
            level_str += f" (+{len(item['job_levels'])-4} more)"

        lines.append(
            f"- Name: {item['name']}\n"
            f"  URL: {item['url']}\n"
            f"  Type: {item['test_type']} ({', '.join(item['keys'])})\n"
            f"  Duration: {item['duration'] or 'Not specified'}\n"
            f"  Languages: {lang_str or 'Not specified'}\n"
            f"  Job Levels: {level_str or 'Not specified'}\n"
            f"  Description: {item['description'][:200]}{'...' if len(item['description']) > 200 else ''}\n"
        )
    return "\n".join(lines)