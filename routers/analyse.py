import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.session_store import get_df, get_cols
from core.deps import check_usage_gate
from core.supabase_client import increment_usage
from core.fig_utils import fig_to_dict as _fig_to_dict
from core.numpy_encoder import numpy_safe, NumpyEncoder
from modules.column_detector import detect_columns, get_data_quality
from modules.ai_analyst import run_ai_analysis, run_ai_analysis_multi
from modules.ai_engine import (
    generate_executive_summary,
    generate_cross_insights_from_findings,
    generate_anomalies_from_findings,
)
from modules.forecaster import (
    prepare_forecast_data, run_forecast,
    get_forecast_metrics, plot_forecast_chart,
)

router = APIRouter()

class AnalyseRequest(BaseModel):
    session_id: str
    selected_analyses: list[str]
    targets: dict[str, float] = {}


def fig_to_dict(fn, *args, **kwargs):
    try:
        return _fig_to_dict(fn(*args, **kwargs))
    except Exception:
        return None


@router.post("/analyse")
async def run_analysis(req: AnalyseRequest, user: dict | None = Depends(check_usage_gate)):
    df = get_df(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found. Please re-upload your file.")

    selected = [s.strip() for s in req.selected_analyses if s and isinstance(s, str)]
    if not selected:
        raise HTTPException(status_code=400, detail="No analysis types selected.")

    cols = get_cols(req.session_id) or detect_columns(df)
    quality = get_data_quality(df)

    sections: dict = {}
    ai_summaries: list[str] = []

    # ── AI engine — run sections in parallel, cache results by file hash ─────
    ai_sections = [s for s in selected if s != "Demand & Forecast"]

    async def _run_section(section: str):
        try:
            return await run_in_threadpool(run_ai_analysis, df, cols, section)
        except Exception:
            return None

    results = await asyncio.gather(*[_run_section(s) for s in ai_sections])
    for ai_result in results:
        if not ai_result:
            continue
        for title, data in ai_result.get('sections', {}).items():
            sections[title] = numpy_safe(data)
        if ai_result.get('executive_summary'):
            ai_summaries.append(ai_result['executive_summary'])

    # ── Demand & Forecast (Prophet — finds best date+numeric cols automatically) ─
    if "Demand & Forecast" in selected:
        try:
            # Auto-detect best date + numeric column if cols don't have them
            if not cols.get('date'):
                for c in df.columns:
                    try:
                        parsed = pd.to_datetime(df[c], errors='coerce')
                        if parsed.notna().mean() > 0.7:
                            cols = {**cols, 'date': c}
                            break
                    except Exception:
                        pass
            if not cols.get('revenue') and not cols.get('quantity'):
                id_hints = ('id', 'index', 'key', 'row', 'seq', 'no', 'num', 'code', 'zip')
                numeric_cols = [
                    c for c in df.select_dtypes(include='number').columns
                    if not any(h in c.lower() for h in id_hints)
                ]
                if numeric_cols:
                    cols = {**cols, 'revenue': numeric_cols[0]}

            demand_df = prepare_forecast_data(df, cols)
            if demand_df is None or len(demand_df) < 8:
                # Not enough time-series data — use AI engine instead
                ai_result = await run_in_threadpool(run_ai_analysis, df, cols, "Demand & Forecast")
                for title, data in ai_result.get('sections', {}).items():
                    sections[title] = numpy_safe(data)
            else:
                _, forecast = await run_in_threadpool(run_forecast, demand_df, 13)
                fc_m = get_forecast_metrics(forecast, demand_df)
                sections["Demand & Forecast"] = {
                    "metrics": numpy_safe(fc_m),
                    "commentary": "",
                    "charts": {
                        "forecast": fig_to_dict(plot_forecast_chart, forecast, demand_df) if forecast is not None else None,
                    },
                }
        except Exception:
            # Fallback to AI engine
            try:
                ai_result = await run_in_threadpool(run_ai_analysis, df, cols, "Demand & Forecast")
                for title, data in ai_result.get('sections', {}).items():
                    sections[title] = numpy_safe(data)
            except Exception:
                pass

    # ── Executive summary, cross insights, anomaly alerts — all in parallel ──
    async def _exec_summary():
        if ai_summaries:
            return ai_summaries[0]
        context = (
            f"Dataset: {quality['total_rows']:,} rows, {quality['total_cols']} columns. "
            f"Quality: {quality['quality_score']}/100. Analyses: {', '.join(selected)}."
        )
        return await run_in_threadpool(generate_executive_summary, context)

    async def _cross_insights():
        try:
            return await run_in_threadpool(generate_cross_insights_from_findings, sections)
        except Exception:
            return []

    async def _anomaly_alerts():
        try:
            return await run_in_threadpool(generate_anomalies_from_findings, sections, quality)
        except Exception:
            return []

    exec_summary, cross_insights, anomaly_alerts = await asyncio.gather(
        _exec_summary(), _cross_insights(), _anomaly_alerts()
    )

    # ── Build chat context from actual AI findings ────────────────────────────
    context_parts = [
        f"Dataset: {quality['total_rows']:,} rows, {quality['total_cols']} columns",
        f"Quality: {quality['quality_score']}/100",
    ]
    for title, data in sections.items():
        insights = (data.get('metrics') or {}).get('ai_insights', [])
        if insights:
            context_parts.append(f"{title}: " + " | ".join(str(f) for f in insights[:3]))
    full_context = "\n".join(context_parts)

    if user:
        increment_usage(user["id"])

    return {
        "executive_summary": exec_summary,
        "quality": numpy_safe(quality),
        "sections": sections,
        "anomaly_alerts": anomaly_alerts,
        "cross_insights": cross_insights,
        "full_context": full_context,
    }


@router.post("/analyse/stream")
async def run_analysis_stream(req: AnalyseRequest, user: dict | None = Depends(check_usage_gate)):
    df = get_df(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found. Please re-upload your file.")

    selected = [s.strip() for s in req.selected_analyses if s and isinstance(s, str)]
    if not selected:
        raise HTTPException(status_code=400, detail="No analysis types selected.")

    cols = get_cols(req.session_id) or detect_columns(df)
    quality = get_data_quality(df)

    async def event_generator():
        sections: dict = {}
        ai_summaries: list[str] = []

        # Status event
        yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing data...'})}\n\n"

        # AI sections — run in parallel, yield each as it completes
        ai_sections = [s for s in selected if s != "Demand & Forecast"]

        tasks = [run_in_threadpool(run_ai_analysis, df, cols, s) for s in ai_sections]
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
            except Exception:
                result = None
            if not result:
                continue
            for title, data in result.get('sections', {}).items():
                safe_data = numpy_safe(data)
                sections[title] = safe_data
                yield f"data: {json.dumps({'type': 'section', 'title': title, 'data': safe_data}, cls=NumpyEncoder)}\n\n"
            if result.get('executive_summary'):
                ai_summaries.append(result['executive_summary'])

        # Demand & Forecast
        if "Demand & Forecast" in selected:
            try:
                demand_df = prepare_forecast_data(df, cols)
                if demand_df is None or len(demand_df) < 8:
                    ai_result = await run_in_threadpool(run_ai_analysis, df, cols, "Demand & Forecast")
                    for title, data in ai_result.get('sections', {}).items():
                        safe_data = numpy_safe(data)
                        sections[title] = safe_data
                        yield f"data: {json.dumps({'type': 'section', 'title': title, 'data': safe_data}, cls=NumpyEncoder)}\n\n"
                else:
                    _, forecast = await run_in_threadpool(run_forecast, demand_df, 13)
                    fc_m = get_forecast_metrics(forecast, demand_df)
                    safe_data = {
                        "metrics": numpy_safe(fc_m),
                        "commentary": "",
                        "charts": {
                            "forecast": fig_to_dict(plot_forecast_chart, forecast, demand_df) if forecast is not None else None,
                        },
                    }
                    sections["Demand & Forecast"] = safe_data
                    yield f"data: {json.dumps({'type': 'section', 'title': 'Demand & Forecast', 'data': safe_data}, cls=NumpyEncoder)}\n\n"
            except Exception:
                try:
                    ai_result = await run_in_threadpool(run_ai_analysis, df, cols, "Demand & Forecast")
                    for title, data in ai_result.get('sections', {}).items():
                        safe_data = numpy_safe(data)
                        sections[title] = safe_data
                        yield f"data: {json.dumps({'type': 'section', 'title': title, 'data': safe_data}, cls=NumpyEncoder)}\n\n"
                except Exception:
                    pass

        # Executive summary, cross insights, anomaly alerts — all in parallel
        async def _exec_summary():
            if ai_summaries:
                return ai_summaries[0]
            context = (
                f"Dataset: {quality['total_rows']:,} rows, {quality['total_cols']} columns. "
                f"Quality: {quality['quality_score']}/100. Analyses: {', '.join(selected)}."
            )
            return await run_in_threadpool(generate_executive_summary, context)

        async def _cross_insights():
            try:
                return await run_in_threadpool(generate_cross_insights_from_findings, sections)
            except Exception:
                return []

        async def _anomaly_alerts():
            try:
                return await run_in_threadpool(generate_anomalies_from_findings, sections, quality)
            except Exception:
                return []

        exec_summary, cross_insights, anomaly_alerts = await asyncio.gather(
            _exec_summary(), _cross_insights(), _anomaly_alerts()
        )

        yield f"data: {json.dumps({'type': 'executive', 'summary': exec_summary, 'quality': numpy_safe(quality)}, cls=NumpyEncoder)}\n\n"
        yield f"data: {json.dumps({'type': 'anomalies', 'data': anomaly_alerts}, cls=NumpyEncoder)}\n\n"
        yield f"data: {json.dumps({'type': 'cross_insights', 'data': cross_insights}, cls=NumpyEncoder)}\n\n"

        # Build full context
        context_parts = [
            f"Dataset: {quality['total_rows']:,} rows, {quality['total_cols']} columns",
            f"Quality: {quality['quality_score']}/100",
        ]
        for title, data in sections.items():
            insights = (data.get('metrics') or {}).get('ai_insights', [])
            if insights:
                context_parts.append(f"{title}: " + " | ".join(str(f) for f in insights[:3]))
        full_context = "\n".join(context_parts)

        if user:
            increment_usage(user["id"])

        yield f"data: {json.dumps({'type': 'done', 'full_context': full_context}, cls=NumpyEncoder)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
