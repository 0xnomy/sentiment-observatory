import base64
import json
import mimetypes
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq

MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"


def _read_image_as_data_uri(image_path: Path) -> Optional[str]:
    if not image_path or not image_path.exists():
        return None
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/png"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_claims_and_facts(text: str) -> List[Dict[str, Any]]:
    """Extract potential claims and facts from text for misinformation analysis."""
    claims = []
    
    # Pattern-based claim detection
    claim_patterns = [
        r"(\w+)\s+(?:claims?|says?|announces?|reveals?)\s+(?:that\s+)?([^.!?]+)",
        r"(?:According to|Based on|Studies show|Research indicates)\s+([^.!?]+)",
        r"(\w+)\s+(?:is|are|was|were)\s+(?:the|a|an)\s+([^.!?]+)",
        r"(?:BREAKING|JUST IN|URGENT|ALERT)\s*:?\s*([^.!?]+)",
    ]
    
    for pattern in claim_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) >= 2:
                source, claim = match.groups()
            else:
                claim = match.group(1)
                source = "unknown"
            
            claims.append({
                "text": claim.strip(),
                "source": source.strip(),
                "confidence": 0.7,
                "type": "claim"
            })
    
    return claims


def _detect_misinformation_patterns(text: str) -> Dict[str, Any]:
    """Detect potential misinformation patterns in text."""
    patterns = {
        "emotional_language": 0,
        "urgency_indicators": 0,
        "conspiracy_indicators": 0,
        "factual_claims": 0,
        "source_quality": 0
    }
    
    # Emotional language detection
    emotional_words = [
        "shocking", "outrageous", "unbelievable", "incredible", "amazing",
        "terrifying", "horrifying", "devastating", "miraculous", "unprecedented"
    ]
    patterns["emotional_language"] = sum(1 for word in emotional_words if word.lower() in text.lower())
    
    # Urgency indicators
    urgency_words = ["now", "immediately", "urgent", "breaking", "just in", "alert", "warning"]
    patterns["urgency_indicators"] = sum(1 for word in urgency_words if word.lower() in text.lower())
    
    # Conspiracy indicators
    conspiracy_words = ["they", "them", "hidden", "secret", "cover-up", "conspiracy", "agenda"]
    patterns["conspiracy_indicators"] = sum(1 for word in conspiracy_words if word.lower() in text.lower())
    
    # Factual claims (numbers, dates, specific details)
    factual_patterns = [
        r"\d+%", r"\d+\s+(?:million|billion|thousand)", r"\d{4}", r"on\s+\w+\s+\d+",
        r"according\s+to\s+\w+", r"study\s+shows", r"research\s+indicates"
    ]
    patterns["factual_claims"] = sum(1 for pattern in factual_patterns if re.search(pattern, text, re.IGNORECASE))
    
    # Source quality (mentions of credible sources)
    credible_sources = ["university", "research", "study", "scientists", "experts", "official"]
    patterns["source_quality"] = sum(1 for source in credible_sources if source.lower() in text.lower())
    
    # Calculate risk score
    risk_factors = [
        patterns["emotional_language"] * 0.3,
        patterns["urgency_indicators"] * 0.4,
        patterns["conspiracy_indicators"] * 0.5,
        -patterns["factual_claims"] * 0.2,
        patterns["source_quality"] * 0.3
    ]
    
    risk_score = min(10, max(1, sum(risk_factors) + 5))
    
    return {
        "risk_score": risk_score,
        "patterns": patterns,
        "claims": _extract_claims_and_facts(text),
        "assessment": "high_risk" if risk_score > 7 else "medium_risk" if risk_score > 4 else "low_risk"
    }


def _categorize_themes(text: str) -> List[Dict[str, Any]]:
    """Categorize content into themes using pattern matching and keyword analysis."""
    themes = []
    
    # Define theme categories with keywords
    theme_keywords = {
        "politics": ["election", "government", "policy", "democrat", "republican", "congress", "senate", "president"],
        "technology": ["ai", "artificial intelligence", "tech", "software", "hardware", "startup", "innovation"],
        "finance": ["stock", "market", "economy", "investment", "crypto", "bitcoin", "ethereum", "trading"],
        "health": ["medical", "health", "vaccine", "covid", "disease", "treatment", "research"],
        "entertainment": ["movie", "music", "celebrity", "film", "actor", "singer", "show"],
        "sports": ["football", "basketball", "soccer", "baseball", "game", "team", "player"],
        "science": ["research", "study", "scientists", "discovery", "experiment", "theory"],
        "environment": ["climate", "environment", "pollution", "sustainability", "green", "carbon"]
    }
    
    text_lower = text.lower()
    
    for theme, keywords in theme_keywords.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if matches > 0:
            confidence = min(1.0, matches / len(keywords) + 0.3)
            themes.append({
                "theme": theme,
                "confidence": confidence,
                "keywords_found": [k for k in keywords if k in text_lower]
            })
    
    # Sort by confidence
    themes.sort(key=lambda x: x["confidence"], reverse=True)
    
    return themes[:5]  # Top 5 themes


def analyze_content_advanced(
    text: str, 
    image_path: Optional[str] = None,
    platform: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Advanced content analysis with theme categorization, misinformation detection,
    and cross-platform correlation analysis.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")

    client = Groq(api_key=api_key)

    # Enhanced system instructions for advanced analysis
    system_instructions = (
        "You are an advanced content analyst specializing in social media content analysis. "
        "Analyze both text and visual content when available. "
        "Output strict JSON only with keys: summary, sentiment, virality_score, report, themes, "
        "content_type, target_audience, engagement_potential, visual_analysis.\n"
        "Constraints:\n"
        "- summary: <= 200 tokens (incorporate both text and visual elements)\n"
        "- sentiment: JSON object {label: positive|neutral|negative, reason: one-line}\n"
        "- virality_score: integer 1-10 with brief rationale\n"
        "- report: bullet points array (<= 300 tokens total)\n"
        "- themes: list of objects [{theme, confidence}] (3-5 items)\n"
        "- content_type: one of [news, opinion, entertainment, educational, promotional, personal]\n"
        "- target_audience: one of [general, specific_demographic, professionals, enthusiasts]\n"
        "- engagement_potential: one of [high, medium, low] with reason\n"
        "- visual_analysis: brief description of visual elements and their impact\n"
        "When analyzing images, consider composition, colors, text overlays, and emotional impact. "
        "Return only JSON with the specified keys."
    )

    user_text_block = (
        "Analyze the content below and return JSON.\n\n"
        "TEXT:\n" + (text or "").strip()[:8000]
    )
    
    if image_path:
        user_text_block += "\n\nIMAGES: Visual content is also provided for analysis. Consider how the visual elements complement or contrast with the text content."

    content_parts = [
        {"type": "text", "text": user_text_block},
    ]

    if image_path:
        data_uri = _read_image_as_data_uri(Path(image_path))
        if data_uri:
            content_parts.append(
                {
                    "type": "input_image",
                    "image_url": data_uri,
                }
            )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": content_parts},
            ],
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content if response.choices else "{}"
        data = json.loads(raw)
    except Exception:
        # Fallback: basic analysis
        data = {
            "summary": text[:500] if text else "",
            "sentiment": {"label": "neutral", "reason": "Analysis failed"},
            "virality_score": 5,
            "report": ["Content analysis completed with fallback methods"],
            "themes": [],
            "content_type": "unknown",
            "target_audience": "general",
            "engagement_potential": "medium",
            "visual_analysis": "Visual analysis not available"
        }

    # Normalize fields
    if isinstance(data.get("sentiment"), str):
        data["sentiment"] = {"label": data["sentiment"], "reason": ""}

    # Ensure report is string for downstream rendering
    rep = data.get("report", [])
    if isinstance(rep, list):
        data["report"] = "\n- " + "\n- ".join([str(x) for x in rep]) if rep else ""
    else:
        data["report"] = str(rep)

    # Coerce virality_score to int
    try:
        data["virality_score"] = int(float(data.get("virality_score", 0)))
    except Exception:
        data["virality_score"] = 0

    # Add advanced analysis components
    data["misinformation_analysis"] = _detect_misinformation_patterns(text)
    data["theme_categorization"] = _categorize_themes(text)
    data["platform_context"] = platform or "unknown"
    data["analysis_timestamp"] = datetime.now().isoformat()
    
    # Add metadata if provided
    if metadata:
        data["source_metadata"] = metadata

    # Constrain field lengths
    data["summary"] = str(data.get("summary", ""))[:4000]
    data["report"] = str(data.get("report", ""))[:6000]
    
    return data


def analyze_cross_platform_correlation(
    content_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze correlation between content across different platforms and time periods.
    """
    if not content_list:
        return {"correlation_score": 0, "common_themes": [], "narrative_consistency": 0}
    
    # Extract themes from all content
    all_themes = []
    for content in content_list:
        themes = content.get("theme_categorization", [])
        all_themes.extend([t["theme"] for t in themes])
    
    # Count theme frequency
    theme_counts = {}
    for theme in all_themes:
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    # Calculate correlation metrics
    total_content = len(content_list)
    common_themes = [theme for theme, count in theme_counts.items() if count > 1]
    
    # Narrative consistency (how many pieces share common themes)
    narrative_consistency = len(common_themes) / max(len(set(all_themes)), 1)
    
    # Sentiment correlation
    sentiments = [c.get("sentiment", {}).get("label", "neutral") for c in content_list]
    sentiment_consistency = len(set(sentiments)) / max(len(sentiments), 1)
    
    # Calculate overall correlation score
    correlation_score = (narrative_consistency + sentiment_consistency) / 2
    
    return {
        "correlation_score": correlation_score,
        "common_themes": common_themes,
        "narrative_consistency": narrative_consistency,
        "sentiment_consistency": sentiment_consistency,
        "total_content_analyzed": total_content,
        "theme_distribution": theme_counts
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python advanced_analyzer.py <text> [image_path] [platform]", file=sys.stderr)
        raise SystemExit(1)
    
    text_arg = sys.argv[1]
    image_arg = sys.argv[2] if len(sys.argv) >= 3 else None
    platform_arg = sys.argv[3] if len(sys.argv) >= 4 else None
    
    result = analyze_content_advanced(text_arg, image_arg, platform_arg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
