import json
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.numpy_encoder import NumpyEncoder
from routers import upload, analyse, chat, nl_chart, report, usage

app = FastAPI(title="Analyst.ai API", version="1.0.0")

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "https://analyst-ai.vercel.app",
        "https://www.analyst.ai",
        "https://analyst.ai",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Numpy-safe JSON response ───────────────────────────────────────────────────
class NumpyJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, cls=NumpyEncoder).encode("utf-8")

app.router.default_response_class = NumpyJSONResponse


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(upload.router,   prefix="/api")
app.include_router(analyse.router,  prefix="/api")
app.include_router(chat.router,     prefix="/api")
app.include_router(nl_chart.router, prefix="/api")
app.include_router(report.router,   prefix="/api")
app.include_router(usage.router,    prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
