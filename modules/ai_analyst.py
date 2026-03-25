"""
Profile → Plan → Execute AI analyst engine.
Fully dynamic — AI chooses sections, metrics, and chart types.
"""
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv

from core.fig_utils import fig_to_dict
from modules.ai_engine import generate_section_commentary

load_dotenv()

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL      = "llama-3.3-70b-versatile"
_FAST_MODEL = "llama-3.1-8b-instant"


_gemini_client = None
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
try:
    from google import genai as _genai
    _gemini_client = _genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
except Exception:
    pass

_cerebras = None
try:
    from cerebras.cloud.sdk import Cerebras
    _cerebras = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY", ""))
except Exception:
    pass


def _gemini_complete(messages: list, max_tokens: int = 1200) -> str:
    if not _gemini_client:
        raise RuntimeError("Gemini not configured")
    from google.genai import types as _gtypes
    user_text = "\n\n".join(m['content'] for m in messages if m['role'] != 'system')
    res = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_text,
        config=_gtypes.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0.1),
    )
    return res.text


def _complete(messages: list, max_tokens: int = 1200, temperature: float = 0.1) -> str:
    """Gemini 3.1 Flash Lite → Cerebras → Groq 70B → Groq 8B."""
    # 1. Gemini — 500 RPD, best for structured JSON
    try:
        result = _gemini_complete(messages, max_tokens=max_tokens)
        print("[Analysis] Using Gemini 3.1 Flash Lite")
        return result
    except Exception as e:
        print(f"[Analysis] Gemini failed ({e}), trying Cerebras")
    # 2. Cerebras — ultra fast, same quality as Groq 70B
    if _cerebras:
        try:
            res = _cerebras.chat.completions.create(
                model="qwen-3-235b-a22b-instruct-2507",
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            print("[Analysis] Using Cerebras llama-3.3-70b")
            return res.choices[0].message.content
        except Exception as e:
            print(f"[Analysis] Cerebras failed ({e}), trying Groq")
    # 3. Groq fallback
    for model in (_MODEL, _FAST_MODEL):
        try:
            res = _groq.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=25.0,
            )
            print(f"[Analysis] Using Groq {model}")
            return res.choices[0].message.content
        except Exception as e:
            if '429' in str(e):
                continue
            raise
    raise RuntimeError("All AI providers rate-limited or unavailable")

LAYOUT = dict(
    paper_bgcolor='#07111f', plot_bgcolor='#07111f',
    font=dict(color='#94a3b8', family='DM Sans', size=12),
    title_font=dict(color='#f1f5f9', family='Syne', size=13),
    xaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f'),
    yaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f'),
    legend=dict(bgcolor='#07111f', bordercolor='#1e3a5f'),
    margin=dict(t=45, b=35, l=45, r=20),
    height=320,
)
COLORS = ['#f59e0b', '#38bdf8', '#10b981', '#ef4444', '#a78bfa', '#fb923c', '#34d399']

CHART_MENU = """
Chart types (pick the best fit):
COMPARISON: bar(x=cat,y=num,agg) | bar_horizontal | grouped_bar(color_by=cat) | radar | bullet
TREND: line(x=date,y=num,agg) | area | area_stacked(color_by=cat) | candlestick(needs ohlc cols)
DISTRIBUTION: histogram(x=num) | box(y=num,x=cat) | violin | strip | density_heatmap(x=num,y=num)
RELATIONSHIP: scatter(x=num,y=num) | bubble(x=num,y=num,size=num) | heatmap | parallel_coords
PART-TO-WHOLE: pie(x=cat,y=num) | donut | treemap | sunburst(color_by=cat) | waterfall | funnel | sankey(x=cat,color_by=cat,y=num)
INDICATORS: gauge(y=num) | indicator(y=num)
STATISTICAL: pareto(x=cat,y=num) | lollipop | slope(color_by=cat) | error_bar | qq_plot(x=num)
OTHER: choropleth(x=country_col,y=num) | pivot_heatmap(x=cat,color_by=cat,y=num) | calendar_heatmap(x=date,y=num) | polar_bar(x=cat,y=num) | timeline(x=cat,start=date,end=date) | dumbell(color_by=cat,needs 2 groups)
"""


# ── Column resolution ─────────────────────────────────────────────────────────
def _resolve_col(name: str | None, df_columns) -> str | None:
    if not name:
        return None
    cols = list(df_columns)
    if name in cols:
        return name
    lower_map = {c.lower(): c for c in cols}
    return lower_map.get(str(name).lower())


# ── Step 1: Profile ───────────────────────────────────────────────────────────
def profile_dataset(df: pd.DataFrame, cols: dict) -> dict:
    sample = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df
    numeric = sample.select_dtypes(include=[np.number])
    categorical = sample.select_dtypes(include=['object'])

    col_profiles = {}
    for col in sample.columns:
        series = sample[col].dropna()
        profile: dict = {
            'dtype': str(sample[col].dtype),
            'null_pct': round(sample[col].isnull().mean() * 100, 1),
            'nunique': int(sample[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(sample[col]):
            profile.update({
                'mean': round(float(series.mean()), 2) if len(series) else None,
                'std':  round(float(series.std()), 2)  if len(series) else None,
                'min':  round(float(series.min()), 2)  if len(series) else None,
                'max':  round(float(series.max()), 2)  if len(series) else None,
                'sum':  round(float(series.sum()), 2)  if len(series) else None,
            })
        else:
            vc = series.value_counts()
            profile['top_values'] = vc.head(5).index.tolist() if len(vc) else []
        col_profiles[col] = profile

    top_corr = []
    if numeric.shape[1] >= 2:
        num_cap = numeric.iloc[:, :15]
        corr = num_cap.corr()
        pairs = []
        cols_list = list(num_cap.columns)
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                v = corr.iloc[i, j]
                if not np.isnan(v):
                    pairs.append((cols_list[i], cols_list[j], round(float(v), 2)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_corr = [{'col_a': a, 'col_b': b, 'r': r} for a, b, r in pairs[:5]]

    return {
        'shape': {'rows': len(df), 'cols': len(df.columns)},
        'numeric_cols': list(numeric.columns),
        'categorical_cols': list(categorical.columns),
        'date_cols': [c for c in sample.columns if 'date' in c.lower() or 'time' in c.lower()],
        'column_profiles': col_profiles,
        'top_correlations': top_corr,
        'detected_roles': {k: v for k, v in cols.items() if v},
    }


# ── Step 2: Plan ──────────────────────────────────────────────────────────────
def plan_analysis(profile: dict, analysis_category: str, df_sample: str, num_sections: int = 4) -> dict | None:
    col_summary = {}
    for col, p in profile['column_profiles'].items():
        if p['dtype'] in ('int64', 'float64') or 'int' in p['dtype'] or 'float' in p['dtype']:
            col_summary[col] = f"numeric | mean={p.get('mean')} min={p.get('min')} max={p.get('max')} sum={p.get('sum')}"
        else:
            col_summary[col] = f"categorical | {p['nunique']} unique | top: {p.get('top_values', [])[:3]}"

    section_rule = f"exactly {num_sections}" if num_sections > 4 else f"2-{num_sections}"

    prompt = f"""You are a senior business analyst and strategic advisor. Analyse this dataset and create a decision-focused report plan.

Dataset: {profile['shape']['rows']:,} rows, {profile['shape']['cols']} columns
Analysis category: {analysis_category}
Detected column roles: {json.dumps(profile['detected_roles'])}

Column profiles:
{json.dumps(col_summary, indent=2)[:2000]}

Top correlations: {json.dumps(profile['top_correlations'])}

Sample rows:
{df_sample[:600]}

{CHART_MENU}

Rules:
1. Create {section_rule} sections with specific titles relevant to THIS dataset
2. Each section must have exactly 3 items — one TREND, one INSIGHT, one RECOMMENDATION:
   - TREND: what is happening, with exact numbers (e.g. "Revenue grew 18% in Q3 vs Q2")
   - INSIGHT: why it matters (e.g. "Top 3 products drive 71% of revenue — bottom 40% contribute under 8%")
   - RECOMMENDATION: a clear business action (e.g. "Increase inventory for top 3 products by 20% ahead of Q4 to avoid stockouts")
3. Recommendations must be specific and actionable — never vague. Say WHAT to do, by HOW MUCH, and WHY.
4. Choose 2-3 charts that best reveal the story — use variety
5. ONLY use column names that EXACTLY match: {list(profile['column_profiles'].keys())}
6. For bubble charts: size= field is required (another numeric column)
7. For treemap/funnel/waterfall: x=categorical, y=numeric
8. Key metrics should use actual computed values from column profiles

Return ONLY valid JSON:
{{
  "sections": [
    {{
      "title": "Specific Section Title",
      "findings": [
        "TREND: specific trend with exact number",
        "INSIGHT: why this matters with specific number",
        "RECOMMENDATION: exact action to take with specific target"
      ],
      "charts": [
        {{"type": "bar", "x": "exact_col", "y": "exact_col", "title": "Chart Title", "agg": "sum", "color_by": null}},
        {{"type": "line", "x": "exact_col", "y": "exact_col", "title": "Chart Title", "agg": "sum", "color_by": null}}
      ],
      "key_metrics": {{"Metric Name": "value with unit"}}
    }}
  ],
  "executive_insight": "2-3 sentence summary covering the most critical finding and the single most important action to take"
}}"""

    try:
        raw = _complete([{"role": "user", "content": prompt}], max_tokens=1000).strip()
        start, end = raw.find('{'), raw.rfind('}')
        if start == -1 or end == -1:
            return None
        return json.loads(raw[start:end + 1])
    except Exception:
        return None


# ── Step 3: Execute ───────────────────────────────────────────────────────────
def _build_chart(spec: dict, df: pd.DataFrame) -> dict | None:
    chart_type = spec.get('type', '').lower()
    title = spec.get('title', '')

    try:
        # ── Heatmap ──────────────────────────────────────────────────────────
        if chart_type == 'heatmap':
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                return None
            cols = numeric.columns[:12]
            corr = numeric[cols].corr().round(2)
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                colorscale=[[0, '#ef4444'], [0.5, '#07111f'], [1, '#10b981']],
                zmin=-1, zmax=1, text=corr.values.round(2),
                texttemplate='%{text}', textfont=dict(size=9),
                colorbar=dict(title=dict(text='r', font=dict(color='#94a3b8')), tickfont=dict(color='#94a3b8')),
            ))
            fig.update_layout(title=title or 'Correlation Matrix', **{**LAYOUT, 'height': 380})
            return fig_to_dict(fig)

        x_col   = _resolve_col(spec.get('x'), df.columns)
        y_col   = _resolve_col(spec.get('y'), df.columns)
        size_col = _resolve_col(spec.get('size'), df.columns)
        color_by = _resolve_col(spec.get('color_by'), df.columns)
        agg = spec.get('agg', 'sum')

        # ── Histogram ────────────────────────────────────────────────────────
        if chart_type == 'histogram':
            if not x_col:
                return None
            fig = go.Figure(go.Histogram(x=df[x_col].dropna(), nbinsx=40, marker_color=COLORS[0], opacity=0.8))
            fig.update_layout(title=title or f'Distribution: {x_col}', **LAYOUT)
            return fig_to_dict(fig)

        # ── Scatter ──────────────────────────────────────────────────────────
        if chart_type == 'scatter':
            if not x_col or not y_col:
                return None
            sample = df[[c for c in [x_col, y_col, color_by] if c]].dropna().sample(min(500, len(df)), random_state=42)
            if color_by and color_by in sample.columns:
                fig = go.Figure()
                for i, cat in enumerate(sample[color_by].unique()[:7]):
                    sub = sample[sample[color_by] == cat]
                    fig.add_trace(go.Scatter(x=sub[x_col], y=sub[y_col], mode='markers',
                                             name=str(cat), marker=dict(color=COLORS[i % len(COLORS)], size=5, opacity=0.7)))
            else:
                fig = go.Figure(go.Scatter(x=sample[x_col], y=sample[y_col], mode='markers',
                                           marker=dict(color=COLORS[1], size=5, opacity=0.6)))
            fig.update_layout(title=title or f'{x_col} vs {y_col}',
                              xaxis=dict(**LAYOUT['xaxis'], title=x_col),
                              yaxis=dict(**LAYOUT['yaxis'], title=y_col),
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Bubble ───────────────────────────────────────────────────────────
        if chart_type == 'bubble':
            if not x_col or not y_col or not size_col:
                return None
            needed = [c for c in [x_col, y_col, size_col] if c]
            sample = df[needed].dropna().sample(min(300, len(df)), random_state=42)
            sizes = sample[size_col]
            sizes_norm = ((sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9) * 40 + 5).tolist()
            fig = go.Figure(go.Scatter(
                x=sample[x_col], y=sample[y_col], mode='markers',
                marker=dict(size=sizes_norm, color=COLORS[0], opacity=0.6,
                            sizemode='diameter', line=dict(color='#07111f', width=1)),
                text=sample[size_col].round(2).astype(str),
            ))
            fig.update_layout(title=title or f'{x_col} vs {y_col} (size={size_col})',
                              xaxis=dict(**LAYOUT['xaxis'], title=x_col),
                              yaxis=dict(**LAYOUT['yaxis'], title=y_col),
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Box ──────────────────────────────────────────────────────────────
        if chart_type == 'box':
            if not y_col:
                return None
            fig = go.Figure()
            if x_col and x_col in df.columns:
                for i, cat in enumerate(df[x_col].dropna().unique()[:10]):
                    sub = df[df[x_col] == cat][y_col].dropna()
                    fig.add_trace(go.Box(y=sub, name=str(cat), marker_color=COLORS[i % len(COLORS)],
                                         fillcolor=f'rgba(245,158,11,0.1)'))
            else:
                fig.add_trace(go.Box(y=df[y_col].dropna(), name=y_col, marker_color=COLORS[0]))
            fig.update_layout(title=title or f'Distribution: {y_col}', **{**LAYOUT, 'height': 340})
            return fig_to_dict(fig)

        # ── Pie ──────────────────────────────────────────────────────────────
        if chart_type == 'pie':
            if not x_col:
                return None
            if y_col and agg in ('sum', 'mean'):
                grp = df.groupby(x_col)[y_col].agg(agg).nlargest(10).reset_index()
                values, labels = grp[y_col], grp[x_col].astype(str)
            else:
                vc = df[x_col].value_counts().head(10)
                labels, values = vc.index.astype(str), vc.values
            fig = go.Figure(go.Pie(labels=labels, values=values,
                                   marker=dict(colors=COLORS, line=dict(color='#07111f', width=1)),
                                   textfont=dict(color='#f1f5f9')))
            fig.update_layout(title=title or f'{x_col} Breakdown',
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Treemap ──────────────────────────────────────────────────────────
        if chart_type == 'treemap':
            if not x_col or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else 'mean'
            grp = df.groupby(x_col)[y_col].agg(fn).reset_index().nlargest(25, y_col)
            fig = go.Figure(go.Treemap(
                labels=grp[x_col].astype(str),
                parents=[''] * len(grp),
                values=grp[y_col],
                marker=dict(colorscale=[[0, '#1e3a5f'], [0.5, '#f59e0b'], [1, '#10b981']],
                            line=dict(color='#07111f', width=1)),
                textfont=dict(color='#f1f5f9'),
            ))
            fig.update_layout(title=title or f'{y_col} by {x_col}',
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Funnel ───────────────────────────────────────────────────────────
        if chart_type == 'funnel':
            if not x_col or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else 'mean'
            grp = df.groupby(x_col)[y_col].agg(fn).reset_index().sort_values(y_col, ascending=False).head(10)
            fig = go.Figure(go.Funnel(
                y=grp[x_col].astype(str), x=grp[y_col],
                marker=dict(color=COLORS[:len(grp)], line=dict(color='#07111f', width=1)),
                textfont=dict(color='#f1f5f9'),
            ))
            fig.update_layout(title=title or f'{y_col} Funnel by {x_col}',
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Waterfall ────────────────────────────────────────────────────────
        if chart_type == 'waterfall':
            if not x_col or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else 'mean'
            grp = df.groupby(x_col)[y_col].agg(fn).reset_index().head(15)
            fig = go.Figure(go.Waterfall(
                x=grp[x_col].astype(str), y=grp[y_col],
                measure=['relative'] * len(grp),
                connector=dict(line=dict(color='#1e3a5f')),
                increasing=dict(marker=dict(color='#10b981')),
                decreasing=dict(marker=dict(color='#ef4444')),
                totals=dict(marker=dict(color='#f59e0b')),
                textfont=dict(color='#f1f5f9'),
            ))
            fig.update_layout(title=title or f'{y_col} Waterfall by {x_col}', **LAYOUT)
            return fig_to_dict(fig)

        # ── Area ─────────────────────────────────────────────────────────────
        if chart_type == 'area':
            if not x_col or not y_col:
                return None
            try:
                df2 = df.copy()
                df2[x_col] = pd.to_datetime(df2[x_col], errors='coerce')
                df2 = df2.dropna(subset=[x_col])
                fn = 'sum' if agg == 'sum' else 'mean'
                grp = df2.groupby(df2[x_col].dt.to_period('M'))[y_col].agg(fn).reset_index()
                grp[x_col] = grp[x_col].astype(str)
            except Exception:
                grp = df[[x_col, y_col]].dropna().head(200)

            if color_by and color_by in df.columns:
                fig = go.Figure()
                for i, cat in enumerate(df[color_by].dropna().unique()[:5]):
                    try:
                        sub = df[df[color_by] == cat].copy()
                        sub[x_col] = pd.to_datetime(sub[x_col], errors='coerce')
                        fn = 'sum' if agg == 'sum' else 'mean'
                        sub = sub.dropna(subset=[x_col]).groupby(sub[x_col].dt.to_period('M'))[y_col].agg(fn).reset_index()
                        sub[x_col] = sub[x_col].astype(str)
                    except Exception:
                        continue
                    fig.add_trace(go.Scatter(x=sub[x_col], y=sub[y_col], mode='lines', name=str(cat),
                                             fill='tozeroy', line=dict(color=COLORS[i % len(COLORS)], width=2)))
            else:
                fig = go.Figure(go.Scatter(x=grp[x_col], y=grp[y_col], mode='lines',
                                           fill='tozeroy', line=dict(color=COLORS[0], width=2)))
            fig.update_layout(title=title or f'{y_col} Area Chart', **LAYOUT)
            return fig_to_dict(fig)

        # ── Bar ──────────────────────────────────────────────────────────────
        if chart_type == 'bar':
            if not x_col or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else ('count' if agg == 'count' else 'mean')
            if agg == 'count':
                grp = df.groupby(x_col).size().reset_index(name=y_col)
            else:
                grp = df.groupby(x_col)[y_col].agg(fn).reset_index()
            grp = grp.dropna().nlargest(20, y_col)
            if color_by and color_by in df.columns:
                fig = go.Figure()
                for i, cat in enumerate(df[color_by].dropna().unique()[:6]):
                    sub = df[df[color_by] == cat].groupby(x_col)[y_col].agg(fn).reset_index()
                    fig.add_trace(go.Bar(x=sub[x_col].astype(str), y=sub[y_col],
                                          name=str(cat), marker_color=COLORS[i % len(COLORS)]))
                fig.update_layout(barmode='group', title=title or f'{y_col} by {x_col}', **LAYOUT)
            else:
                fig = go.Figure(go.Bar(x=grp[x_col].astype(str), y=grp[y_col],
                                       marker_color=COLORS[0], marker_line_color='#07111f', marker_line_width=1))
                fig.update_layout(title=title or f'{y_col} by {x_col}', **LAYOUT)
            return fig_to_dict(fig)

        # ── Line ─────────────────────────────────────────────────────────────
        if chart_type == 'line':
            if not x_col or not y_col:
                return None
            try:
                df2 = df.copy()
                df2[x_col] = pd.to_datetime(df2[x_col], errors='coerce')
                df2 = df2.dropna(subset=[x_col])
                fn = 'sum' if agg == 'sum' else 'mean'
                grp = df2.groupby(df2[x_col].dt.to_period('M'))[y_col].agg(fn).reset_index()
                grp[x_col] = grp[x_col].astype(str)
            except Exception:
                grp = df[[x_col, y_col]].dropna().head(200)
            if color_by and color_by in df.columns:
                fig = go.Figure()
                for i, cat in enumerate(df[color_by].dropna().unique()[:6]):
                    try:
                        sub = df[df[color_by] == cat].copy()
                        sub[x_col] = pd.to_datetime(sub[x_col], errors='coerce')
                        fn = 'sum' if agg == 'sum' else 'mean'
                        sub = sub.dropna(subset=[x_col]).groupby(sub[x_col].dt.to_period('M'))[y_col].agg(fn).reset_index()
                        sub[x_col] = sub[x_col].astype(str)
                    except Exception:
                        continue
                    fig.add_trace(go.Scatter(x=sub[x_col], y=sub[y_col], mode='lines+markers',
                                              name=str(cat), line=dict(color=COLORS[i % len(COLORS)], width=2)))
            else:
                fig = go.Figure(go.Scatter(x=grp[x_col], y=grp[y_col], mode='lines+markers',
                                           line=dict(color=COLORS[0], width=2), marker=dict(size=4)))
            fig.update_layout(title=title or f'{y_col} over time', **LAYOUT)
            return fig_to_dict(fig)

        # ── Bar Horizontal ───────────────────────────────────────────────────
        if chart_type == 'bar_horizontal':
            if not x_col or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else ('count' if agg == 'count' else 'mean')
            grp = df.groupby(x_col)[y_col].agg(fn).reset_index().nlargest(20, y_col)
            fig = go.Figure(go.Bar(
                y=grp[x_col].astype(str), x=grp[y_col],
                orientation='h', marker_color=COLORS[0],
                marker_line_color='#07111f', marker_line_width=1,
            ))
            fig.update_layout(title=title or f'{y_col} by {x_col}',
                              yaxis=dict(**LAYOUT['yaxis'], autorange='reversed'),
                              **{k: v for k, v in LAYOUT.items() if k != 'yaxis'})
            return fig_to_dict(fig)

        # ── Grouped Bar (alias) ──────────────────────────────────────────────
        if chart_type == 'grouped_bar':
            spec2 = {**spec, 'type': 'bar'}
            return _build_chart(spec2, df)

        # ── Violin ──────────────────────────────────────────────────────────
        if chart_type == 'violin':
            if not y_col:
                return None
            fig = go.Figure()
            if x_col and x_col in df.columns:
                for i, cat in enumerate(df[x_col].dropna().unique()[:8]):
                    sub = df[df[x_col] == cat][y_col].dropna()
                    fig.add_trace(go.Violin(y=sub, name=str(cat),
                                            box_visible=True, meanline_visible=True,
                                            line_color=COLORS[i % len(COLORS)],
                                            fillcolor=f'rgba({int(COLORS[i%len(COLORS)][1:3],16)},{int(COLORS[i%len(COLORS)][3:5],16)},{int(COLORS[i%len(COLORS)][5:],16)},0.15)'))
            else:
                fig.add_trace(go.Violin(y=df[y_col].dropna(), name=y_col,
                                         box_visible=True, meanline_visible=True,
                                         line_color=COLORS[0], fillcolor='rgba(245,158,11,0.1)'))
            fig.update_layout(title=title or f'Distribution: {y_col}', **{**LAYOUT, 'height': 340})
            return fig_to_dict(fig)

        # ── Strip plot ───────────────────────────────────────────────────────
        if chart_type == 'strip':
            if not y_col:
                return None
            sample = df.sample(min(500, len(df)), random_state=42)
            fig = go.Figure()
            if x_col and x_col in df.columns:
                for i, cat in enumerate(sample[x_col].dropna().unique()[:8]):
                    sub = sample[sample[x_col] == cat][y_col].dropna()
                    fig.add_trace(go.Box(y=sub, name=str(cat), boxpoints='all', jitter=0.4,
                                          pointpos=0, marker=dict(color=COLORS[i % len(COLORS)], size=3),
                                          line=dict(color='rgba(0,0,0,0)'), fillcolor='rgba(0,0,0,0)'))
            else:
                fig.add_trace(go.Box(y=sample[y_col].dropna(), boxpoints='all', jitter=0.4,
                                      pointpos=0, marker=dict(color=COLORS[0], size=3),
                                      line=dict(color='rgba(0,0,0,0)'), fillcolor='rgba(0,0,0,0)'))
            fig.update_layout(title=title or f'Strip: {y_col}', **LAYOUT)
            return fig_to_dict(fig)

        # ── Density Heatmap ──────────────────────────────────────────────────
        if chart_type == 'density_heatmap':
            if not x_col or not y_col:
                return None
            sample = df[[x_col, y_col]].dropna().sample(min(2000, len(df)), random_state=42)
            fig = go.Figure(go.Histogram2dContour(
                x=sample[x_col], y=sample[y_col],
                colorscale=[[0, '#07111f'], [0.5, '#1e3a5f'], [1, '#f59e0b']],
                contours=dict(coloring='heatmap'),
                line=dict(width=0),
            ))
            fig.update_layout(title=title or f'Density: {x_col} vs {y_col}',
                              xaxis=dict(**LAYOUT['xaxis'], title=x_col),
                              yaxis=dict(**LAYOUT['yaxis'], title=y_col),
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Donut ────────────────────────────────────────────────────────────
        if chart_type == 'donut':
            spec2 = {**spec, 'type': 'pie'}
            result = _build_chart(spec2, df)
            if result and result.get('data'):
                for trace in result['data']:
                    trace['hole'] = 0.45
            return result

        # ── Sunburst ─────────────────────────────────────────────────────────
        if chart_type == 'sunburst':
            if not x_col or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else 'mean'
            if color_by and color_by in df.columns:
                grp = df.groupby([color_by, x_col])[y_col].agg(fn).reset_index()
                parents = grp[color_by].astype(str).tolist()
                labels = grp[x_col].astype(str).tolist()
                values = grp[y_col].tolist()
                # add parent nodes
                parent_totals = grp.groupby(color_by)[y_col].sum().reset_index()
                labels = list(parent_totals[color_by].astype(str)) + labels
                values = list(parent_totals[y_col]) + values
                parents = [''] * len(parent_totals) + parents
            else:
                grp = df.groupby(x_col)[y_col].agg(fn).reset_index().nlargest(20, y_col)
                labels = grp[x_col].astype(str).tolist()
                values = grp[y_col].tolist()
                parents = [''] * len(grp)
            fig = go.Figure(go.Sunburst(
                labels=labels, parents=parents, values=values,
                marker=dict(colorscale=[[0, '#1e3a5f'], [0.5, '#f59e0b'], [1, '#10b981']]),
                textfont=dict(color='#f1f5f9'),
            ))
            fig.update_layout(title=title or f'{y_col} Hierarchy',
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Sankey ───────────────────────────────────────────────────────────
        if chart_type == 'sankey':
            if not x_col or not color_by or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else 'mean'
            grp = df.groupby([x_col, color_by])[y_col].agg(fn).reset_index()
            sources = grp[x_col].astype(str).tolist()
            targets = grp[color_by].astype(str).tolist()
            all_nodes = list(dict.fromkeys(sources + targets))
            node_idx = {n: i for i, n in enumerate(all_nodes)}
            fig = go.Figure(go.Sankey(
                node=dict(label=all_nodes,
                          color=COLORS[:len(all_nodes)] if len(all_nodes) <= len(COLORS) else COLORS * (len(all_nodes) // len(COLORS) + 1),
                          pad=15, thickness=20, line=dict(color='#07111f', width=0.5)),
                link=dict(source=[node_idx[s] for s in sources],
                          target=[node_idx[t] for t in targets],
                          value=grp[y_col].tolist(),
                          color='rgba(245,158,11,0.2)'),
            ))
            fig.update_layout(title=title or f'Flow: {x_col} → {color_by}',
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Radar ────────────────────────────────────────────────────────────
        if chart_type == 'radar':
            if not x_col or not y_col:
                return None
            fn = 'mean'
            grp = df.groupby(x_col)[y_col].agg(fn).reset_index().head(8)
            cats = grp[x_col].astype(str).tolist()
            vals = grp[y_col].tolist()
            cats_closed = cats + [cats[0]]
            vals_closed = vals + [vals[0]]
            fig = go.Figure(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill='toself', line=dict(color=COLORS[0], width=2),
                fillcolor='rgba(245,158,11,0.1)',
            ))
            fig.update_layout(title=title or f'{y_col} Radar',
                              polar=dict(
                                  bgcolor='#07111f',
                                  radialaxis=dict(gridcolor='#1e3a5f', color='#94a3b8'),
                                  angularaxis=dict(gridcolor='#1e3a5f', color='#94a3b8'),
                              ),
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Gauge ────────────────────────────────────────────────────────────
        if chart_type == 'gauge':
            if not y_col:
                return None
            val = float(df[y_col].mean())
            max_val = float(df[y_col].max())
            fig = go.Figure(go.Indicator(
                mode='gauge+number',
                value=val,
                gauge=dict(
                    axis=dict(range=[0, max_val], tickcolor='#94a3b8'),
                    bar=dict(color='#f59e0b'),
                    bgcolor='#07111f',
                    bordercolor='#1e3a5f',
                    steps=[
                        dict(range=[0, max_val * 0.5], color='#0a1628'),
                        dict(range=[max_val * 0.5, max_val * 0.75], color='#1e3a5f'),
                        dict(range=[max_val * 0.75, max_val], color='#1e2a3f'),
                    ],
                    threshold=dict(line=dict(color='#10b981', width=2), thickness=0.75, value=val),
                ),
                number=dict(font=dict(color='#f1f5f9')),
                title=dict(text=title or f'Average {y_col}', font=dict(color='#94a3b8')),
            ))
            fig.update_layout(**{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Indicator (big number) ───────────────────────────────────────────
        if chart_type == 'indicator':
            if not y_col:
                return None
            val = float(df[y_col].sum() if agg == 'sum' else df[y_col].mean())
            fig = go.Figure(go.Indicator(
                mode='number+delta',
                value=val,
                number=dict(font=dict(color='#f59e0b', size=48)),
                title=dict(text=title or y_col, font=dict(color='#94a3b8', size=14)),
            ))
            fig.update_layout(**{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Parallel Coordinates ─────────────────────────────────────────────
        if chart_type == 'parallel_coords':
            numeric = df.select_dtypes(include=[np.number]).iloc[:, :10].dropna()
            if numeric.shape[1] < 3:
                return None
            sample = numeric.sample(min(500, len(numeric)), random_state=42)
            dims = [dict(range=[sample[c].min(), sample[c].max()],
                         label=c, values=sample[c]) for c in sample.columns]
            fig = go.Figure(go.Parcoords(
                line=dict(color=sample.iloc[:, 0],
                          colorscale=[[0, '#1e3a5f'], [0.5, '#f59e0b'], [1, '#10b981']]),
                dimensions=dims,
                labelfont=dict(color='#94a3b8'),
                tickfont=dict(color='#94a3b8'),
            ))
            fig.update_layout(title=title or 'Parallel Coordinates',
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Area Stacked (alias of area with color_by) ───────────────────────
        if chart_type == 'area_stacked':
            spec2 = {**spec, 'type': 'area'}
            return _build_chart(spec2, df)

        # ── Bullet chart ─────────────────────────────────────────────────────
        if chart_type == 'bullet':
            if not x_col or not y_col:
                return None
            fn = 'mean'
            grp = df.groupby(x_col)[y_col].agg(fn).reset_index().head(10)
            overall_mean = grp[y_col].mean()
            fig = go.Figure()
            for i, row in grp.iterrows():
                fig.add_trace(go.Bar(
                    x=[row[y_col]], y=[str(row[x_col])],
                    orientation='h',
                    marker_color=COLORS[0] if row[y_col] >= overall_mean else COLORS[3],
                    name=str(row[x_col]),
                ))
                fig.add_shape(type='line',
                              x0=overall_mean, x1=overall_mean,
                              y0=i - 0.4, y1=i + 0.4,
                              line=dict(color='#38bdf8', width=2))
            fig.update_layout(title=title or f'{y_col} vs Average',
                              showlegend=False, barmode='overlay',
                              yaxis=dict(**LAYOUT['yaxis'], autorange='reversed'),
                              **{k: v for k, v in LAYOUT.items() if k != 'yaxis'})
            return fig_to_dict(fig)

        # ── Candlestick ──────────────────────────────────────────────────────
        if chart_type == 'candlestick':
            open_col  = _resolve_col(spec.get('open')  or 'open',  df.columns)
            high_col  = _resolve_col(spec.get('high')  or 'high',  df.columns)
            low_col   = _resolve_col(spec.get('low')   or 'low',   df.columns)
            close_col = _resolve_col(spec.get('close') or 'close', df.columns)
            date_col  = x_col
            if not all([date_col, open_col, high_col, low_col, close_col]):
                return None
            df2 = df[[date_col, open_col, high_col, low_col, close_col]].dropna()
            df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
            df2 = df2.dropna().sort_values(date_col)
            fig = go.Figure(go.Candlestick(
                x=df2[date_col], open=df2[open_col],
                high=df2[high_col], low=df2[low_col], close=df2[close_col],
                increasing=dict(line=dict(color='#10b981'), fillcolor='rgba(16,185,129,0.3)'),
                decreasing=dict(line=dict(color='#ef4444'), fillcolor='rgba(239,68,68,0.3)'),
            ))
            fig.update_layout(title=title or 'Price Chart',
                              xaxis=dict(**LAYOUT['xaxis'], rangeslider=dict(visible=False)),
                              yaxis=LAYOUT['yaxis'],
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Error Bar ────────────────────────────────────────────────────────
        if chart_type == 'error_bar':
            if not x_col or not y_col:
                return None
            grp = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index().head(20)
            fig = go.Figure(go.Bar(
                x=grp[x_col].astype(str), y=grp['mean'],
                error_y=dict(type='data', array=grp['std'].fillna(0).tolist(), visible=True,
                             color='#38bdf8', thickness=1.5),
                marker_color=COLORS[0], marker_line_color='#07111f', marker_line_width=1,
            ))
            fig.update_layout(title=title or f'{y_col} Mean ± Std by {x_col}', **LAYOUT)
            return fig_to_dict(fig)

        # ── QQ Plot ──────────────────────────────────────────────────────────
        if chart_type == 'qq_plot':
            if not x_col:
                return None
            from scipy import stats as scipy_stats
            data = df[x_col].dropna().sample(min(500, len(df)), random_state=42)
            (osm, osr), (slope, intercept, _) = scipy_stats.probplot(data)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(osm), y=list(osr), mode='markers',
                                     marker=dict(color=COLORS[0], size=4, opacity=0.7), name='Data'))
            line_x = [min(osm), max(osm)]
            fig.add_trace(go.Scatter(x=line_x, y=[slope * x + intercept for x in line_x],
                                     mode='lines', line=dict(color='#38bdf8', width=2), name='Normal'))
            fig.update_layout(title=title or f'Q-Q Plot: {x_col}',
                              xaxis=dict(**LAYOUT['xaxis'], title='Theoretical Quantiles'),
                              yaxis=dict(**LAYOUT['yaxis'], title='Sample Quantiles'),
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Pareto ───────────────────────────────────────────────────────────
        if chart_type == 'pareto':
            if not x_col or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else 'mean'
            grp = df.groupby(x_col)[y_col].agg(fn).reset_index().sort_values(y_col, ascending=False).head(20)
            grp['cum_pct'] = grp[y_col].cumsum() / grp[y_col].sum() * 100
            fig = go.Figure()
            fig.add_trace(go.Bar(x=grp[x_col].astype(str), y=grp[y_col],
                                  marker_color=COLORS[0], name=y_col))
            fig.add_trace(go.Scatter(x=grp[x_col].astype(str), y=grp['cum_pct'],
                                      mode='lines+markers', yaxis='y2',
                                      line=dict(color='#38bdf8', width=2), name='Cumulative %'))
            fig.update_layout(
                title=title or f'Pareto: {y_col}',
                yaxis2=dict(overlaying='y', side='right', range=[0, 105],
                            ticksuffix='%', gridcolor='#1e3a5f', color='#94a3b8'),
                **LAYOUT,
            )
            return fig_to_dict(fig)

        # ── Lollipop ─────────────────────────────────────────────────────────
        if chart_type == 'lollipop':
            if not x_col or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else 'mean'
            grp = df.groupby(x_col)[y_col].agg(fn).reset_index().nlargest(20, y_col)
            fig = go.Figure()
            for _, row in grp.iterrows():
                fig.add_shape(type='line', x0=str(row[x_col]), x1=str(row[x_col]),
                              y0=0, y1=row[y_col], line=dict(color='#1e3a5f', width=1.5))
            fig.add_trace(go.Scatter(x=grp[x_col].astype(str), y=grp[y_col],
                                      mode='markers', marker=dict(color=COLORS[0], size=10),
                                      name=y_col))
            fig.update_layout(title=title or f'{y_col} by {x_col}', **LAYOUT)
            return fig_to_dict(fig)

        # ── Slope Chart ──────────────────────────────────────────────────────
        if chart_type == 'slope':
            if not x_col or not y_col or not color_by:
                return None
            fn = 'mean'
            grp = df.groupby([color_by, x_col])[y_col].agg(fn).reset_index()
            fig = go.Figure()
            for i, cat in enumerate(grp[color_by].unique()[:10]):
                sub = grp[grp[color_by] == cat].sort_values(x_col)
                fig.add_trace(go.Scatter(x=sub[x_col].astype(str), y=sub[y_col],
                                          mode='lines+markers', name=str(cat),
                                          line=dict(color=COLORS[i % len(COLORS)], width=2),
                                          marker=dict(size=8)))
            fig.update_layout(title=title or f'{y_col} Slope by {color_by}', **LAYOUT)
            return fig_to_dict(fig)

        # ── Choropleth ───────────────────────────────────────────────────────
        if chart_type == 'choropleth':
            if not x_col or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else 'mean'
            grp = df.groupby(x_col)[y_col].agg(fn).reset_index()
            fig = go.Figure(go.Choropleth(
                locations=grp[x_col].astype(str),
                z=grp[y_col],
                locationmode='country names',
                colorscale=[[0, '#07111f'], [0.5, '#1e3a5f'], [1, '#f59e0b']],
                colorbar=dict(tickfont=dict(color='#94a3b8')),
            ))
            fig.update_layout(
                title=title or f'{y_col} by Country',
                geo=dict(bgcolor='#07111f', showframe=False,
                         showcoastlines=True, coastlinecolor='#1e3a5f',
                         landcolor='#0a1628', showland=True),
                **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')},
            )
            return fig_to_dict(fig)

        # ── Pivot Heatmap ────────────────────────────────────────────────────
        if chart_type in ('pivot_heatmap', 'annotated_heatmap'):
            if not x_col or not color_by or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else 'mean'
            pivot = df.groupby([x_col, color_by])[y_col].agg(fn).unstack(fill_value=0)
            pivot = pivot.iloc[:20, :20]
            text = pivot.round(1).values if chart_type == 'annotated_heatmap' else None
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.astype(str).tolist(),
                y=pivot.index.astype(str).tolist(),
                colorscale=[[0, '#07111f'], [0.5, '#1e3a5f'], [1, '#f59e0b']],
                text=text, texttemplate='%{text}' if text is not None else None,
                textfont=dict(size=8),
                colorbar=dict(tickfont=dict(color='#94a3b8')),
            ))
            fig.update_layout(title=title or f'{y_col} by {x_col} × {color_by}',
                              **{**LAYOUT, 'height': 380})
            return fig_to_dict(fig)

        # ── Contour ──────────────────────────────────────────────────────────
        if chart_type == 'contour':
            if not x_col or not y_col:
                return None
            sample = df[[x_col, y_col]].dropna().sample(min(1000, len(df)), random_state=42)
            fig = go.Figure(go.Histogram2dContour(
                x=sample[x_col], y=sample[y_col],
                colorscale=[[0, '#07111f'], [0.5, '#1e3a5f'], [1, '#10b981']],
                contours=dict(coloring='heatmap', showlabels=True,
                              labelfont=dict(color='#f1f5f9', size=9)),
            ))
            fig.update_layout(title=title or f'Contour: {x_col} vs {y_col}',
                              xaxis=dict(**LAYOUT['xaxis'], title=x_col),
                              yaxis=dict(**LAYOUT['yaxis'], title=y_col),
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Polar Bar ────────────────────────────────────────────────────────
        if chart_type == 'polar_bar':
            if not x_col or not y_col:
                return None
            fn = 'sum' if agg == 'sum' else 'mean'
            grp = df.groupby(x_col)[y_col].agg(fn).reset_index().head(12)
            fig = go.Figure(go.Barpolar(
                r=grp[y_col], theta=grp[x_col].astype(str),
                marker=dict(color=COLORS[:len(grp)], line=dict(color='#07111f', width=1)),
            ))
            fig.update_layout(
                title=title or f'{y_col} by {x_col} (Polar)',
                polar=dict(bgcolor='#07111f',
                           radialaxis=dict(gridcolor='#1e3a5f', color='#94a3b8'),
                           angularaxis=dict(gridcolor='#1e3a5f', color='#94a3b8')),
                **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')},
            )
            return fig_to_dict(fig)

        # ── OHLC ─────────────────────────────────────────────────────────────
        if chart_type == 'ohlc':
            spec2 = {**spec, 'type': 'candlestick'}
            return _build_chart(spec2, df)

        # ── Timeline / Gantt ─────────────────────────────────────────────────
        if chart_type == 'timeline':
            start_col = _resolve_col(spec.get('start') or spec.get('x'), df.columns)
            end_col   = _resolve_col(spec.get('end') or spec.get('y'), df.columns)
            label_col = _resolve_col(spec.get('color_by') or x_col, df.columns)
            if not start_col or not end_col or not label_col:
                return None
            df2 = df[[label_col, start_col, end_col]].dropna().head(30)
            df2[start_col] = pd.to_datetime(df2[start_col], errors='coerce')
            df2[end_col]   = pd.to_datetime(df2[end_col],   errors='coerce')
            df2 = df2.dropna()
            fig = go.Figure()
            for i, row in df2.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row[start_col], row[end_col]],
                    y=[str(row[label_col]), str(row[label_col])],
                    mode='lines', line=dict(color=COLORS[i % len(COLORS)], width=10),
                    name=str(row[label_col]), showlegend=False,
                ))
            fig.update_layout(title=title or 'Timeline',
                              xaxis=dict(**LAYOUT['xaxis'], type='date'),
                              yaxis=LAYOUT['yaxis'],
                              **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')})
            return fig_to_dict(fig)

        # ── Dumbell ──────────────────────────────────────────────────────────
        if chart_type == 'dumbell':
            if not x_col or not y_col or not color_by:
                return None
            fn = 'mean'
            groups = df[color_by].dropna().unique()[:2]
            if len(groups) < 2:
                return None
            g1 = df[df[color_by] == groups[0]].groupby(x_col)[y_col].agg(fn).reset_index()
            g2 = df[df[color_by] == groups[1]].groupby(x_col)[y_col].agg(fn).reset_index()
            merged = g1.merge(g2, on=x_col, suffixes=('_a', '_b')).head(20)
            fig = go.Figure()
            for _, row in merged.iterrows():
                fig.add_shape(type='line',
                              x0=row[f'{y_col}_a'], x1=row[f'{y_col}_b'],
                              y0=str(row[x_col]), y1=str(row[x_col]),
                              line=dict(color='#1e3a5f', width=2))
            fig.add_trace(go.Scatter(x=merged[f'{y_col}_a'], y=merged[x_col].astype(str),
                                      mode='markers', name=str(groups[0]),
                                      marker=dict(color=COLORS[0], size=10)))
            fig.add_trace(go.Scatter(x=merged[f'{y_col}_b'], y=merged[x_col].astype(str),
                                      mode='markers', name=str(groups[1]),
                                      marker=dict(color=COLORS[2], size=10)))
            fig.update_layout(title=title or f'{y_col}: {groups[0]} vs {groups[1]}',
                              **{k: v for k, v in LAYOUT.items()})
            return fig_to_dict(fig)

        # ── Calendar Heatmap ─────────────────────────────────────────────────
        if chart_type == 'calendar_heatmap':
            if not x_col or not y_col:
                return None
            try:
                df2 = df[[x_col, y_col]].copy()
                df2[x_col] = pd.to_datetime(df2[x_col], errors='coerce')
                df2 = df2.dropna()
                fn = 'sum' if agg == 'sum' else 'mean'
                daily = df2.groupby(df2[x_col].dt.date)[y_col].agg(fn).reset_index()
                daily.columns = ['date', 'value']
                daily['date'] = pd.to_datetime(daily['date'])
                daily['week'] = daily['date'].dt.isocalendar().week.astype(int)
                daily['dow'] = daily['date'].dt.dayofweek
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig = go.Figure(go.Heatmap(
                    x=daily['week'], y=daily['dow'],
                    z=daily['value'],
                    colorscale=[[0, '#07111f'], [0.5, '#1e3a5f'], [1, '#f59e0b']],
                    colorbar=dict(tickfont=dict(color='#94a3b8')),
                ))
                fig.update_layout(
                    title=title or f'{y_col} Calendar',
                    yaxis=dict(**LAYOUT['yaxis'], tickvals=list(range(7)), ticktext=days),
                    xaxis=dict(**LAYOUT['xaxis'], title='Week of Year'),
                    **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')},
                )
                return fig_to_dict(fig)
            except Exception:
                return None

    except Exception:
        return None

    return None


def execute_plan(plan: dict, df: pd.DataFrame) -> dict:
    sections = {}
    for section_spec in plan.get('sections', []):
        title = section_spec.get('title', 'Analysis')
        findings = section_spec.get('findings', [])
        chart_specs = section_spec.get('charts', [])
        key_metrics = section_spec.get('key_metrics', {})

        charts = {}
        for i, spec in enumerate(chart_specs):
            chart = _build_chart(spec, df)
            if chart:
                key = f"chart_{i}_{spec.get('type', 'unknown')}"
                charts[key] = chart

        sections[title] = {
            'metrics': {'ai_insights': findings, **{k: v for k, v in key_metrics.items()}},
            'commentary': '',
            'charts': charts,
        }
    return sections


def _compute_time_metrics(df: pd.DataFrame, cols: dict) -> dict:
    """Compute MTD/QTD/YTD/MoM/YoY from detected date + revenue columns."""
    date_col = cols.get('date')
    rev_col = cols.get('revenue') or cols.get('quantity')
    if not date_col or not rev_col:
        return {}
    try:
        df2 = df[[date_col, rev_col]].copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
        df2[rev_col] = pd.to_numeric(df2[rev_col], errors='coerce')
        df2 = df2.dropna()
        if len(df2) < 2:
            return {}
        today = df2[date_col].max()

        mtd = float(df2[df2[date_col].dt.month == today.month][rev_col].sum())
        qtd = float(df2[df2[date_col].dt.quarter == today.quarter][rev_col].sum())
        ytd = float(df2[df2[date_col].dt.year == today.year][rev_col].sum())

        curr_m = df2[df2[date_col].dt.to_period('M') == today.to_period('M')][rev_col].sum()
        prev_m_date = today - pd.DateOffset(months=1)
        prev_m = df2[df2[date_col].dt.to_period('M') == prev_m_date.to_period('M')][rev_col].sum()
        mom_pct = float((curr_m - prev_m) / prev_m * 100) if prev_m > 0 else 0.0

        curr_y = df2[df2[date_col].dt.year == today.year][rev_col].sum()
        prev_y = df2[df2[date_col].dt.year == today.year - 1][rev_col].sum()
        yoy_pct = float((curr_y - prev_y) / prev_y * 100) if prev_y > 0 else 0.0

        rolling_4w = float(df2[df2[date_col] >= today - pd.Timedelta(weeks=4)][rev_col].sum())

        return {
            'mtd': round(mtd, 2),
            'qtd': round(qtd, 2),
            'ytd': round(ytd, 2),
            'mom_pct': round(mom_pct, 1),
            'yoy_pct': round(yoy_pct, 1),
            'rolling_4w': round(rolling_4w, 2),
        }
    except Exception:
        return {}


# ── Data usability check ──────────────────────────────────────────────────────
def _check_data_usability(df: pd.DataFrame) -> str | None:
    """Return an error message if the data is not suitable for analysis, else None."""
    rows, col_count = len(df), len(df.columns)

    # Too few rows
    if rows < 5:
        return f"Only {rows} rows found — not enough data for meaningful analysis. Please upload a dataset with at least 10 rows."

    # All columns are text/object — no numeric data to analyse
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    if col_count <= 3 and not numeric_cols:
        # Check if this looks like a data dictionary (FIELDS + DESCRIPTION pattern)
        col_names_lower = [c.lower() for c in df.columns]
        is_dictionary = any(
            any(kw in name for kw in ('field', 'column', 'attribute', 'variable', 'name'))
            for name in col_names_lower
        ) and any(
            any(kw in name for kw in ('description', 'desc', 'definition', 'type', 'meaning'))
            for name in col_names_lower
        )
        if is_dictionary:
            return (
                f"This file appears to be a data dictionary or schema document ({rows} rows, "
                f"columns: {', '.join(df.columns.tolist())}), not a dataset. "
                "Please upload your actual data file (transactions, orders, customers, etc.) "
                "to run an analysis."
            )

    # Very few columns, all text, high uniqueness — looks like a list or log, not structured data
    if col_count == 1 and not numeric_cols:
        return (
            "Only 1 text column found. Please upload a structured dataset with multiple "
            "columns (e.g. CSV with headers) to enable analysis."
        )

    # Enough rows and at least some structure — OK to analyse
    return None


# ── Result cache (in-memory, keyed by file hash + categories) ─────────────────
import hashlib, functools
_result_cache: dict = {}
_CACHE_MAX = 50  # keep last 50 results

def _file_hash(df: pd.DataFrame) -> str:
    """Fast hash of dataframe content for cache keying."""
    try:
        sample = pd.util.hash_pandas_object(df.head(1000), index=False).sum()
        return f"{len(df)}_{len(df.columns)}_{sample}"
    except Exception:
        return f"{len(df)}_{len(df.columns)}"


# ── Multi-category entry point (single AI call for all categories) ────────────
def run_ai_analysis_multi(df: pd.DataFrame, cols: dict, categories: list[str]) -> dict:
    """Profile once, plan all categories in one AI call, execute once."""
    issue = _check_data_usability(df)
    if issue:
        return {
            'sections': {'Data Notice': {'metrics': {'ai_insights': [issue]}, 'commentary': '', 'charts': {}}},
            'executive_summary': issue,
            'ai_driven': True,
            'data_issue': True,
        }

    # Cache check
    cache_key = _file_hash(df) + "|" + ",".join(sorted(categories))
    if cache_key in _result_cache:
        print(f"[Cache] Hit for {cache_key[:40]}")
        return _result_cache[cache_key]

    profile = profile_dataset(df, cols)
    df_sample = df.head(5).to_string(index=False, max_cols=15)[:600]

    # Single plan call covering all categories — scale sections with category count
    combined_category = " + ".join(categories) if len(categories) > 1 else categories[0]
    num_sections = min(2 * len(categories) + 2, 6)
    plan = plan_analysis(profile, combined_category, df_sample, num_sections=num_sections)
    if not plan:
        raise ValueError("AI planning step failed")

    df_chart = df.sample(50_000, random_state=42) if len(df) > 50_000 else df
    sections = execute_plan(plan, df_chart)

    # Inject time metrics if any executive/dashboard category
    if any(any(kw in cat.lower() for kw in ('executive', 'dashboard')) for cat in categories):
        tm = _compute_time_metrics(df, cols)
        if tm:
            for title, section_data in sections.items():
                if any(kw in title.lower() for kw in ('executive', 'dashboard', 'kpi', 'performance', 'overview')):
                    section_data['time_metrics'] = tm
                    break

    for section_data in sections.values():
        section_data['commentary'] = ''

    result = {
        'sections': sections,
        'executive_summary': plan.get('executive_insight', ''),
        'ai_driven': True,
    }

    # Store in cache, evict oldest if over limit
    if len(_result_cache) >= _CACHE_MAX:
        oldest = next(iter(_result_cache))
        del _result_cache[oldest]
    _result_cache[cache_key] = result
    return result


# ── Main entry point ──────────────────────────────────────────────────────────
def run_ai_analysis(df: pd.DataFrame, cols: dict, analysis_category: str = 'Auto') -> dict:
    # Gate: reject files that aren't suitable for analysis
    issue = _check_data_usability(df)
    if issue:
        return {
            'sections': {
                'Data Notice': {
                    'metrics': {'ai_insights': [issue]},
                    'commentary': '',
                    'charts': {},
                }
            },
            'executive_summary': issue,
            'ai_driven': True,
            'data_issue': True,
        }

    # Cache check
    cache_key = _file_hash(df) + "|" + analysis_category
    if cache_key in _result_cache:
        print(f"[Cache] Hit: {analysis_category}")
        return _result_cache[cache_key]

    profile = profile_dataset(df, cols)
    df_sample = df.head(5).to_string(index=False, max_cols=15)[:600]
    plan = plan_analysis(profile, analysis_category, df_sample)
    if not plan:
        raise ValueError("AI planning step failed")

    df_chart = df.sample(50_000, random_state=42) if len(df) > 50_000 else df
    sections = execute_plan(plan, df_chart)

    # Inject time metrics into executive/dashboard sections
    if any(kw in analysis_category.lower() for kw in ('executive', 'dashboard')):
        tm = _compute_time_metrics(df, cols)
        if tm:
            for title, section_data in sections.items():
                if any(kw in title.lower() for kw in ('executive', 'dashboard', 'kpi', 'performance', 'overview')):
                    section_data['time_metrics'] = tm
                    break

    full_context = json.dumps({
        'rows': profile['shape']['rows'],
        'category': analysis_category,
        'correlations': profile['top_correlations'],
    })[:800]

    for section_data in sections.values():
        section_data['commentary'] = ''  # AI findings in metrics.ai_insights serve as commentary

    result = {
        'sections': sections,
        'executive_summary': plan.get('executive_insight', ''),
        'ai_driven': True,
    }

    if len(_result_cache) >= _CACHE_MAX:
        del _result_cache[next(iter(_result_cache))]
    _result_cache[cache_key] = result
    return result
