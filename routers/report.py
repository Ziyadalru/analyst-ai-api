from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from modules.report_generator import generate_report, _chart_to_png

router = APIRouter()


class ReportRequest(BaseModel):
    executive_summary: str = ""
    quality: dict = {}
    sections: dict = {}
    selected_analyses: list[str] = []


@router.get("/report/test-chart")
async def test_chart():
    import traceback
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar(['A','B','C'], [1,2,3])
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        fig.savefig(tmp.name)
        plt.close(fig)
        size = os.path.getsize(tmp.name)
        os.unlink(tmp.name)
        return {"status": "ok", "matplotlib": matplotlib.__version__, "png_bytes": size}
    except Exception as e:
        return {"status": "error", "error": str(e), "trace": traceback.format_exc()}


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
