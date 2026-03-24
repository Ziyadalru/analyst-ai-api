from fastapi import APIRouter
from pydantic import BaseModel
from core.session_store import get_df
from core.fig_utils import fig_to_dict
from modules.ai_engine import chat_response, _analysis_complete
from modules.nl_chart import natural_language_chart, build_df_schema

router = APIRouter()

_CHART_KEYWORDS = {
    'chart', 'graph', 'plot', 'show me', 'visualize', 'visualise',
    'bar chart', 'line chart', 'pie chart', 'scatter', 'histogram',
    'draw', 'create a chart', 'build a chart', 'make a chart',
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    messages: list[ChatMessage]
    full_context: str = ""
    use_pro: bool = False


def _wants_chart(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in _CHART_KEYWORDS)



_NO_INTENT_PHRASES = {
    'i want a chart', 'show me a chart', 'give me a chart', 'make a chart',
    'create a chart', 'a chart', 'some charts', 'chart please', 'chart',
    'show me a graph', 'give me a graph', 'i want a graph',
}

def _has_intent(query: str) -> bool:
    """True if the user specified a topic (delivery, profit, etc.)."""
    t = query.lower().strip()
    return t not in _NO_INTENT_PHRASES and len(t.split()) > 3 or any(
        kw in t for kw in ('deliver', 'profit', 'revenue', 'sales', 'customer',
                           'product', 'region', 'ship', 'order', 'category',
                           'margin', 'cost', 'late', 'trend', 'segment')
    )


def _resolve_chart_query(user_query: str, df, full_context: str) -> str:
    """Make a vague query specific. If no intent, pick a new insight not in the analysis."""
    schema = build_df_schema(df)
    col_names = list(schema.keys())[:30]
    has_intent = _has_intent(user_query)

    if has_intent:
        instruction = (
            "The user wants a chart about a specific topic but hasn't given enough detail. "
            "Make their request specific using exact column names from the list. "
            "Respect their topic — don't change the subject. "
            "Format: 'show [chart_type] of [metric] by [dimension]'. Return ONLY the chart request."
        )
        context_hint = f"User's topic: '{user_query}'"
    else:
        instruction = (
            "The user wants any chart. Suggest ONE that reveals a NEW insight "
            "not already covered in the analysis. "
            "Format: 'show [chart_type] of [metric] by [dimension]' using exact column names. "
            "Return ONLY the chart request string."
        )
        context_hint = f"Analysis already covers: {full_context[:800]}"

    try:
        result = _analysis_complete([
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Available columns: {col_names}\n\n{context_hint}"}
        ], max_tokens=60, temperature=0.2)
        return result.strip().strip('"\'')
    except Exception:
        return user_query


def _build_data_context(df) -> str:
    if df is None:
        return ""
    cols = {col: str(dtype) for col, dtype in df.dtypes.items()}
    try:
        sample = df.head(5).to_string(index=False, max_cols=20)
    except Exception:
        sample = ""
    return (
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
        f"Columns & types: {cols}\n"
        f"Sample rows:\n{sample}"
    )


@router.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    df = get_df(req.session_id)
    data_context = _build_data_context(df)

    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    # Try to generate chart first so we know whether it succeeded
    chart = None
    if _wants_chart(last_user) and df is not None:
        try:
            # Try the user's exact query first
            print(f"[Chart] Generating chart for: {last_user!r}")
            fig, explanation = natural_language_chart(last_user, df)
            # If spec failed (too vague / unrecognised), let AI pick a new insight
            if fig is None:
                resolved = _resolve_chart_query(last_user, df, req.full_context)
                print(f"[Chart] Resolved to: {resolved!r}")
                fig, explanation = natural_language_chart(resolved, df)
            print(f"[Chart] fig={fig is not None}, explanation={explanation!r}")
            if fig is not None:
                chart = fig_to_dict(fig)
        except Exception as e:
            print(f"[Chart] Exception: {e}")
    elif _wants_chart(last_user) and df is None:
        print(f"[Chart] SKIPPED — df is None for session {req.session_id!r}")

    # Frame the AI message based on actual chart outcome
    if _wants_chart(last_user):
        if chart is not None:
            framed = messages[:-1] + [{
                "role": "user",
                "content": f"{last_user} (Chart is rendered. Give the key business insight in 2–3 sentences. No code.)"
            }]
        else:
            framed = messages[:-1] + [{
                "role": "user",
                "content": f"{last_user} (No chart could be generated — the data may not be loaded. Give the key insight from the analysis results as plain text instead.)"
            }]
        reply = chat_response(framed, req.full_context, data_context, use_pro=req.use_pro)
    else:
        reply = chat_response(messages, req.full_context, data_context, use_pro=req.use_pro)

    return {"reply": reply, "chart": chart}
