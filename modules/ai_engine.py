import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Groq ─────────────────────────────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
FAST_MODEL = "llama-3.1-8b-instant"

# ── Cerebras (faster than Groq, same models) ──────────────────────────────────
_cerebras = None
try:
    from cerebras.cloud.sdk import Cerebras
    _cerebras = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY", ""))
except Exception:
    pass

# ── Gemini (free tier) ────────────────────────────────────────────────────────
_gemini_client = None
GEMINI_MODEL      = "gemini-3.1-flash-lite-preview"  # analysis: 500 RPD
GEMINI_CHAT_MODELS = [                               # chat: quality cascade
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
]
GEMMA_CHAT_MODEL = "gemma-3-27b-it"
try:
    from google import genai as _genai
    _gemini_client = _genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
except Exception:
    pass

# ── Claude (pro tier) ────────────────────────────────────────────────────────
_claude = None
CLAUDE_MODEL = "claude-sonnet-4-6"
try:
    import anthropic as _anthropic
    _claude = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
except Exception:
    pass


def _gemini_complete(messages: list, max_tokens: int = 500) -> str:
    """Call Gemini 3.1 Flash Lite with OpenAI-style messages list."""
    if not _gemini_client:
        raise RuntimeError("Gemini not configured")
    from google.genai import types as _gtypes
    system_parts = [m['content'] for m in messages if m['role'] == 'system']
    user_parts   = [m['content'] for m in messages if m['role'] != 'system']
    res = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents="\n\n".join(user_parts),
        config=_gtypes.GenerateContentConfig(
            system_instruction=system_parts[0] if system_parts else None,
            max_output_tokens=max_tokens,
            temperature=0.2,
        ),
    )
    return res.text


# Alias — both planning and secondary calls use same model now
_gemini_complete_lite = _gemini_complete


# ── Shared system prompt for chat ────────────────────────────────────────────
def _chat_system(full_context: str, data_context: str) -> str:
    parts = [
        "You are Analyst.ai — a senior business analyst with deep expertise in finance, "
        "supply chain, customer analytics, operations, and risk management.",
        "Answer like a McKinsey consultant: specific, data-driven, direct. Use actual numbers from the data.",
        "Give actionable recommendations, not just observations. Keep answers to 3–5 sentences.",
        "Never start a response with 'No' or any negation. Never hedge with phrases like 'if it were' — use the real data.",
        "You are embedded in a data analytics platform. Charts are rendered automatically — you never write code.",
        "When charts are shown, give the key business insight in 2–3 plain sentences.",
        "NEVER use markdown: no **bold**, no headers, no bullet points, no ``` code blocks. Plain prose only.",
        "NEVER write Python, SQL, or any code.",
        "Industry benchmarks: OTIF >85%, profit margin varies by industry, late delivery <5% is excellent.",
    ]
    if data_context:
        parts.append(f"\nDataset:\n{data_context}")
    if full_context:
        parts.append(f"\nAnalysis results:\n{full_context[:1500]}")
    return "\n".join(parts)


# ── Cerebras ─────────────────────────────────────────────────────────────────
def _cerebras_complete(messages: list, max_tokens: int = 1200, temperature: float = 0.3) -> str:
    if not _cerebras:
        raise RuntimeError("Cerebras not configured")
    res = _cerebras.chat.completions.create(
        model="qwen-3-235b-a22b-instruct-2507",
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    return res.choices[0].message.content


# ── Groq chat ────────────────────────────────────────────────────────────────
def _groq_chat(messages: list, full_context: str, data_context: str) -> str:
    system = _chat_system(full_context, data_context)
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=0.3,
            max_tokens=1200,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"I encountered an error: {e}"


# ── Claude chat ───────────────────────────────────────────────────────────────
def _claude_chat(messages: list, full_context: str, data_context: str) -> str:
    if not _claude:
        return _groq_chat(messages, full_context, data_context)
    system = _chat_system(full_context, data_context)
    try:
        res = _claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1200,
            system=system,
            messages=messages,
        )
        return res.content[0].text
    except Exception as e:
        return _groq_chat(messages, full_context, data_context)


def _gemini_chat_with_model(model: str, messages: list, full_context: str, data_context: str) -> str:
    if not _gemini_client:
        raise RuntimeError("Gemini not configured")
    from google.genai import types as _gtypes
    system = _chat_system(full_context, data_context)
    # Build proper multi-turn conversation (user/model alternating turns)
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(_gtypes.Content(
            role=role,
            parts=[_gtypes.Part.from_text(text=m["content"])],
        ))
    res = _gemini_client.models.generate_content(
        model=model,
        contents=contents,
        config=_gtypes.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1200,
            temperature=0.3,
        ),
    )
    return res.text


def _gemma_chat(messages: list, full_context: str, data_context: str) -> str:
    system = _chat_system(full_context, data_context)
    res = client.chat.completions.create(
        model=GEMMA_CHAT_MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.3,
        max_tokens=1200,
    )
    return res.choices[0].message.content


def chat_response(messages: list, full_context: str, data_context: str = "", use_pro: bool = False) -> str:
    if use_pro and _claude:
        return _claude_chat(messages, full_context, data_context)

    # Cerebras — 235B model, fastest and most reliable
    try:
        system = _chat_system(full_context, data_context)
        result = _cerebras_complete([{"role": "system", "content": system}] + messages)
        print("[Chat] Using Cerebras qwen-3-235b")
        return result
    except Exception as e:
        print(f"[Chat] Cerebras failed ({e}), trying Groq")

    # Groq 70B
    try:
        result = _groq_chat(messages, full_context, data_context)
        print("[Chat] Using Groq 70B")
        return result
    except Exception as e:
        print(f"[Chat] Groq 70B failed ({e}), trying Gemini")

    # Gemini cascade as fallback
    for model in GEMINI_CHAT_MODELS:
        try:
            result = _gemini_chat_with_model(model, messages, full_context, data_context)
            print(f"[Chat] Using {model}")
            return result
        except Exception as e:
            print(f"[Chat] {model} failed ({e}), trying next")

    # Gemma 27B
    try:
        result = _gemma_chat(messages, full_context, data_context)
        print("[Chat] Using Gemma 3 27B")
        return result
    except Exception as e:
        print(f"[Chat] Gemma failed ({e}), trying Groq 8B")

    # Groq 8B last resort
    try:
        res = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "system", "content": _chat_system(full_context, data_context)}] + messages,
            temperature=0.3,
            max_tokens=1200,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"I encountered an error: {e}"


# ── Analysis AI (unchanged) ───────────────────────────────────────────────────
def _analysis_complete(messages: list, max_tokens: int = 200, temperature: float = 0.2) -> str:
    """Gemini 2.0 Flash-Lite → Groq 70B → Groq 8B (analysis pipeline)."""
    try:
        result = _gemini_complete_lite(messages, max_tokens=max_tokens)
        print("[AI] Using Gemini 3.1 Flash Lite")
        return result
    except Exception as e:
        print(f"[AI] Gemini Flash-Lite failed ({e}), falling back to Groq")
    for model in (MODEL, FAST_MODEL):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=20.0,
            )
            return res.choices[0].message.content
        except Exception as e:
            if '429' in str(e):
                continue
            raise
    raise RuntimeError("All AI providers rate-limited or unavailable")


def generate_executive_summary(data_context: str) -> str:
    try:
        return _analysis_complete([
            {
                "role": "system",
                "content": (
                    "You are a McKinsey senior partner presenting to a board. "
                    "Write exactly 4 sentences: "
                    "(1) Overall business health with the most important number. "
                    "(2) The single biggest opportunity with a specific action and expected impact. "
                    "(3) The single biggest risk with a specific mitigation step. "
                    "(4) One strategic priority for the next 90 days. "
                    "Use actual numbers from the data. No bullet points. No headers. Plain sentences only."
                )
            },
            {"role": "user", "content": f"Dataset analysis:\n{data_context[:1500]}"}
        ], max_tokens=200, temperature=0.2)
    except Exception as e:
        return f"Executive summary unavailable — AI service error: {e}"


def generate_section_commentary(section: str, metrics: dict, context: str) -> str:
    prompts = {
        'Financial': (
            "You are a CFO. Give exactly 3 sentences: "
            "(1) The most critical financial finding with the exact number and vs industry benchmark. "
            "(2) Root cause in one sentence. "
            "(3) One immediate action with an expected dollar or percentage outcome. "
            "Benchmarks: healthy profit margin 10-20%, MoM growth target >3%, discount >20% signals margin risk."
        ),
        'Supply Chain': (
            "You are a supply chain VP. Give exactly 3 sentences: "
            "(1) The most critical supply chain metric vs SCOR benchmark (OTIF target >95%, late delivery <5%). "
            "(2) Likely root cause. "
            "(3) One specific corrective action with a measurable target and timeline."
        ),
        'Customer': (
            "You are a Chief Customer Officer. Give exactly 3 sentences: "
            "(1) The most important customer finding (CLV, churn risk, segment concentration) with numbers. "
            "(2) What it means for revenue at risk or growth potential. "
            "(3) One retention or expansion action with a specific target segment."
        ),
        'Risk': (
            "You are a Chief Risk Officer. Give exactly 3 sentences: "
            "(1) The highest-priority risk with its probability and financial exposure estimate. "
            "(2) Which other business areas this risk is most likely to impact. "
            "(3) One mitigation action with a specific deadline and owner role."
        ),
        'Data': (
            "You are a Chief Data Officer. Give exactly 3 sentences: "
            "(1) The most important data quality or statistical finding and its business impact. "
            "(2) Which decisions or analyses this data issue would distort if unaddressed. "
            "(3) One remediation step with an estimated effort level (hours/days)."
        ),
        'Executive Dashboard': (
            "You are a management consultant. Give exactly 3 sentences: "
            "(1) The single most important business performance finding this period vs prior period. "
            "(2) Whether the business is on track, at risk, or off track, and why. "
            "(3) One strategic decision the leadership team should make in the next 30 days."
        ),
        'Demand & Forecast': (
            "You are a VP of Demand Planning. Give exactly 3 sentences: "
            "(1) Interpret the forecast trend with the specific growth or decline rate. "
            "(2) The inventory or capacity implication of this trend. "
            "(3) One procurement or production recommendation with a specific quantity or timing."
        ),
    }
    system_prompt = prompts.get(section, "Give exactly 3 sentences: key finding with numbers, root cause, one specific action with measurable outcome.")
    try:
        return _analysis_complete([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Metrics:\n{metrics}\n\nContext summary:\n{context[:800]}"}
        ], max_tokens=140, temperature=0.2)
    except Exception as e:
        return f"Commentary unavailable — AI service error: {e}"


def generate_cross_insights_from_findings(sections: dict) -> list:
    """Generate cross-section insights from actual AI findings."""
    if not sections:
        return []
    findings_text = []
    for title, data in sections.items():
        insights = (data.get('metrics') or {}).get('ai_insights', [])
        if insights:
            findings_text.append(f"[{title}]: " + " | ".join(str(f) for f in insights[:3]))
    if not findings_text:
        return []
    combined = "\n".join(findings_text)
    try:
        raw = _analysis_complete([
            {
                "role": "system",
                "content": (
                    "You are a senior analyst. Given findings from multiple analysis sections, "
                    "identify exactly 3 cross-cutting insights that connect patterns across sections. "
                    "Each insight must reference at least 2 sections by name, use specific numbers, "
                    "and end with one actionable recommendation. "
                    "Format: exactly 3 lines, each starting with [Section1 → Section2] insight. "
                    "No numbering, no extra text."
                )
            },
            {"role": "user", "content": combined}
        ], max_tokens=240, temperature=0.2).strip()
        return [l.strip() for l in raw.split('\n') if l.strip()][:3]
    except Exception:
        return []


def generate_anomalies_from_findings(sections: dict, quality: dict) -> list:
    """Generate AI-driven anomaly alerts from actual section findings."""
    findings_text = []
    for title, data in sections.items():
        insights = (data.get('metrics') or {}).get('ai_insights', [])
        for f in insights:
            findings_text.append(f"[{title}] {f}")
    if not findings_text:
        return []
    combined = "\n".join(findings_text[:15])
    quality_note = f"Data quality: {quality.get('quality_score', 0)}/100, {quality.get('overall_null_pct', 0)}% missing"
    try:
        raw = _analysis_complete([
            {
                "role": "system",
                "content": (
                    "You are a risk analyst. From these findings, identify up to 4 anomalies or risks "
                    "that need immediate attention. Only flag genuinely unusual or concerning patterns. "
                    "Format: JSON array of objects with 'type' (warning/error/info), 'title', 'message'. "
                    "Return ONLY the JSON array."
                )
            },
            {"role": "user", "content": f"{quality_note}\n\nFindings:\n{combined}"}
        ], max_tokens=400, temperature=0.1).strip()
        start, end = raw.find('['), raw.rfind(']')
        if start == -1:
            return []
        import json as _json
        return _json.loads(raw[start:end + 1])
    except Exception:
        return []


def generate_cross_section_insights(context: str) -> list:
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior business analyst. Identify exactly 3 cross-functional insights "
                        "that connect patterns across different business areas. "
                        "Each insight must reference at least 2 different business areas, use specific numbers, "
                        "and end with one actionable recommendation. "
                        "Format: exactly 3 lines, each starting with [Area1 → Area2] Your insight. "
                        "No extra text, no numbering."
                    )
                },
                {"role": "user", "content": f"Full analysis data:\n{context}"}
            ],
            temperature=0.3,
            max_tokens=240,
        )
        raw = response.choices[0].message.content.strip()
        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        return lines[:3] if lines else []
    except Exception:
        return []
