import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq


def score_virality(text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")

    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.2,
        max_tokens=300,
        groq_api_key=api_key,
    )

    parser = JsonOutputParser()
    format_instructions = parser.get_format_instructions()

    template = (
        "You are a concise virality assessor.\n"
        "Score the text for virality on 1-10 based on: emotional intensity, controversy level, shareability.\n"
        "Respond with JSON only. No prose.\n"
        "{format_instructions}\n\n"
        "Text (<= 1500 chars):\n{content}"
    )

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | parser

    content = (text or "").strip()
    if len(content) > 1500:
        content = content[:1500]

    try:
        result = chain.invoke({"content": content, "format_instructions": format_instructions})
    except OutputParserException:
        # Fallback: call LLM directly and try to extract JSON substring
        raw = llm.invoke(prompt.format(content=content, format_instructions=format_instructions))
        text_out = getattr(raw, "content", str(raw))
        import re, json as _json

        match = re.search(r"\{[\s\S]*\}", text_out)
        if match:
            try:
                result = _json.loads(match.group(0))
            except Exception:
                result = {"virality_score": 5, "reason": "fallback"}
        else:
            result = {"virality_score": 5, "reason": "fallback"}

    # Basic normalization
    try:
        score = int(result.get("virality_score", 0))
    except Exception:
        score = 0
    reason = result.get("reason", "")
    return {"virality_score": max(1, min(10, score)), "reason": reason}


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python virality_agent.py <text>", file=sys.stderr)
        raise SystemExit(1)
    text = sys.argv[1]
    print(json.dumps(score_virality(text), ensure_ascii=False))


