from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.session_store import get_df
from core.fig_utils import fig_to_dict
from modules.nl_chart import natural_language_chart

router = APIRouter()


class NLChartRequest(BaseModel):
    session_id: str
    query: str


@router.post("/nl-chart")
async def build_nl_chart(req: NLChartRequest):
    df = get_df(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found. Please re-upload your CSV.")

    try:
        fig, explanation = natural_language_chart(req.query, df)
    except Exception as e:
        return {"chart": None, "explanation": "", "error": str(e)}

    return {
        "chart": fig_to_dict(fig),
        "explanation": explanation or "",
        "error": None,
    }
