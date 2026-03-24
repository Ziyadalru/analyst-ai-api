from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from modules.report_generator import generate_report

router = APIRouter()


class ReportRequest(BaseModel):
    executive_summary: str = ""
    quality: dict = {}
    sections: dict = {}
    selected_analyses: list[str] = []


@router.post("/report")
async def create_report(req: ReportRequest):
    path = generate_report(
        executive_summary=req.executive_summary,
        quality_report=req.quality,
        sections=req.sections,
        selected_analyses=req.selected_analyses,
    )
    return FileResponse(
        path=path,
        media_type="application/pdf",
        filename="analyst_ai_report.pdf",
        background=None,
    )
