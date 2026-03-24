from fastapi import APIRouter
from pydantic import BaseModel
from core.session_store import get_df
from core.fig_utils import fig_to_dict
from modules.ai_engine import chat_response
from modules.nl_chart import natural_language_chart

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

    reply = chat_response(messages, req.full_context, data_context, use_pro=req.use_pro)

    # Generate chart if the user asked for one
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    chart = None
    if _wants_chart(last_user) and df is not None:
        try:
            fig, _ = natural_language_chart(last_user, df)
            chart = fig_to_dict(fig)
        except Exception:
            pass

    return {"reply": reply, "chart": chart}
