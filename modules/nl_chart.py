import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from modules.ai_engine import client, MODEL

LAYOUT = dict(
    paper_bgcolor='#07111f', plot_bgcolor='#07111f',
    font=dict(color='#94a3b8', family='DM Sans', size=12),
    title_font=dict(color='#f1f5f9', family='Syne', size=13),
    xaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f'),
    yaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f'),
    legend=dict(bgcolor='#07111f', bordercolor='#1e3a5f'),
    margin=dict(t=45, b=35, l=45, r=20),
    height=380,
)

COLORS = ['#f59e0b', '#38bdf8', '#10b981', '#a78bfa', '#ef4444', '#f97316', '#06b6d4', '#84cc16']

def get_chart_spec(user_query: str, df_schema: dict) -> dict:
    # Only send first 30 columns to keep prompt size manageable
    trimmed = dict(list(df_schema.items())[:30])
    schema_str = json.dumps(trimmed, indent=2)
    col_list = ', '.join(f'"{c}"' for c in df_schema.keys())
    prompt = f"""You are a data visualization expert. The user wants to create a chart from their dataset.

Available columns (use EXACT names): {col_list}

Dataset schema:
{schema_str}

User request: "{user_query}"

Return ONLY a valid JSON object - no explanation, no markdown:
{{
  "chart_type": "bar|line|scatter|pie|histogram|box|area|heatmap",
  "x": "exact_column_name or null",
  "y": "exact_column_name or null",
  "color": "exact_column_name or null",
  "aggregation": "sum|mean|count|max|min|none",
  "group_by": "exact_column_name or null",
  "top_n": 10,
  "sort": "desc|asc|none",
  "title": "descriptive chart title",
  "x_label": "x axis label",
  "y_label": "y axis label",
  "filters": {{}}
}}

Rules:
- ONLY use column names from the Available columns list above — exact spelling and case
- revenue/sales → use line chart over time or bar by category
- distribution → histogram
- compare → bar chart
- breakdown/share → pie chart
- correlation → scatter
- aggregation applies when group_by is set
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace('```json', '').replace('```', '').strip()
        start, end = raw.find('{'), raw.rfind('}')
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
        return json.loads(raw)
    except Exception:
        return None

def build_df_schema(df: pd.DataFrame) -> dict:
    schema = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if 'datetime' in dtype or 'date' in dtype.lower():
            sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ''
            schema[col] = {'type': 'datetime', 'sample': sample}
        elif df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            schema[col] = {
                'type': 'numeric',
                'min': round(float(df[col].min()), 2),
                'max': round(float(df[col].max()), 2),
                'mean': round(float(df[col].mean()), 2),
            }
        else:
            top_vals = df[col].value_counts().head(5).index.tolist()
            schema[col] = {
                'type': 'categorical',
                'unique': int(df[col].nunique()),
                'top_values': [str(v) for v in top_vals],
            }
    return schema

def execute_chart_spec(df: pd.DataFrame, spec: dict) -> go.Figure:
    if not spec:
        return None
    chart_type = spec.get('chart_type', 'bar')
    x_col = spec.get('x')
    y_col = spec.get('y')
    color_col = spec.get('color')
    group_by = spec.get('group_by')
    agg = spec.get('aggregation', 'sum')
    top_n = spec.get('top_n', 10) or 10
    sort = spec.get('sort', 'desc')
    title = spec.get('title', 'Chart')
    x_label = spec.get('x_label', x_col or '')
    y_label = spec.get('y_label', y_col or '')

    all_cols = df.columns.tolist()
    col_lower = {c.lower(): c for c in all_cols}

    def resolve(name):
        if name is None: return None
        if name in all_cols: return name
        return col_lower.get(str(name).lower())

    x_col = resolve(x_col)
    y_col = resolve(y_col)
    color_col = resolve(color_col)
    group_by = resolve(group_by) or x_col

    df2 = df.copy()
    if x_col:
        try:
            parsed = pd.to_datetime(df2[x_col], errors='coerce')
            if not parsed.isna().all():
                df2[x_col] = parsed
        except:
            pass

    plot_df = df2
    if group_by and y_col and agg != 'none':
        agg_func = {'sum': 'sum', 'mean': 'mean', 'count': 'count', 'max': 'max', 'min': 'min'}.get(agg, 'sum')
        if pd.api.types.is_datetime64_any_dtype(df2[group_by]):
            df2['_period'] = df2[group_by].dt.to_period('M').astype(str)
            grp_col = '_period'
        else:
            grp_col = group_by
        if color_col and color_col != group_by:
            plot_df = df2.groupby([grp_col, color_col])[y_col].agg(agg_func).reset_index()
        else:
            plot_df = df2.groupby(grp_col)[y_col].agg(agg_func).reset_index()
        if sort == 'desc':
            plot_df = plot_df.sort_values(y_col, ascending=False)
        elif sort == 'asc':
            plot_df = plot_df.sort_values(y_col, ascending=True)
        if not pd.api.types.is_datetime64_any_dtype(df2[group_by]):
            plot_df = plot_df.head(int(top_n))
        group_by = grp_col

    x_data = plot_df[group_by] if group_by and group_by in plot_df.columns else (plot_df[x_col] if x_col and x_col in plot_df.columns else None)
    y_data = plot_df[y_col] if y_col and y_col in plot_df.columns else None

    fig = None
    try:
        if chart_type == 'bar':
            if color_col and color_col in plot_df.columns:
                fig = px.bar(plot_df, x=group_by or x_col, y=y_col, color=color_col,
                             color_discrete_sequence=COLORS, title=title)
            else:
                fig = go.Figure(go.Bar(
                    x=x_data, y=y_data, marker_color=COLORS[0],
                    text=[f'{v:,.0f}' if isinstance(v, (int, float)) else str(v) for v in (y_data if y_data is not None else [])],
                    textposition='outside',
                ))
        elif chart_type in ('line', 'area'):
            if color_col and color_col in plot_df.columns:
                fig = px.line(plot_df, x=group_by or x_col, y=y_col, color=color_col,
                              color_discrete_sequence=COLORS, title=title)
            else:
                fig = go.Figure(go.Scatter(
                    x=x_data, y=y_data, mode='lines+markers',
                    line=dict(color=COLORS[0], width=2),
                    fill='tozeroy' if chart_type == 'area' else None,
                    fillcolor='rgba(245,158,11,0.08)' if chart_type == 'area' else None,
                ))
        elif chart_type == 'scatter':
            fig = go.Figure(go.Scatter(
                x=x_data, y=y_data, mode='markers',
                marker=dict(color=COLORS[0], size=6, opacity=0.7),
            ))
        elif chart_type == 'pie':
            if x_data is None or y_data is None:
                chart_type = 'bar'
                fig = go.Figure(go.Bar(x=x_data, y=y_data, marker_color=COLORS[0]))
            else:
                fig = go.Figure(go.Pie(
                    labels=x_data, values=y_data, hole=0.45,
                    marker=dict(colors=COLORS), textinfo='percent+label',
                ))
        elif chart_type == 'histogram':
            target = y_col or x_col
            if target and target in df2.columns:
                fig = go.Figure(go.Histogram(
                    x=df2[target].dropna(), nbinsx=40,
                    marker_color=COLORS[0], opacity=0.8,
                ))
        elif chart_type == 'box':
            target = y_col or x_col
            if target and target in df2.columns:
                if color_col and color_col in df2.columns:
                    fig = px.box(df2, x=color_col, y=target, color=color_col,
                                 color_discrete_sequence=COLORS)
                else:
                    fig = go.Figure(go.Box(
                        y=df2[target].dropna(), marker_color=COLORS[0],
                        fillcolor='rgba(245,158,11,0.1)', line_color=COLORS[0],
                    ))
        elif chart_type == 'heatmap':
            if group_by and color_col and y_col and color_col in df2.columns:
                try:
                    pivot = df2.pivot_table(values=y_col, index=group_by, columns=color_col,
                                            aggfunc='sum', fill_value=0)
                    pivot = pivot.iloc[:15, :15]
                    fig = go.Figure(go.Heatmap(
                        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                        colorscale=[[0, '#07111f'], [0.5, '#1e3a5f'], [1, '#f59e0b']],
                        colorbar=dict(tickfont=dict(color='#94a3b8')),
                    ))
                except:
                    pass

        if fig is None and x_col and y_col:
            fig = go.Figure(go.Bar(x=x_data, y=y_data, marker_color=COLORS[0]))

        if fig:
            fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, **LAYOUT)

    except Exception as e:
        return None

    return fig

def _default_chart(df: pd.DataFrame):
    """Generate a sensible default chart when the query is too vague."""
    schema = build_df_schema(df)
    # Find a numeric col and a categorical col automatically
    num_cols = [c for c, v in schema.items() if v.get('type') == 'numeric']
    cat_cols = [c for c, v in schema.items() if v.get('type') == 'categorical' and v.get('unique', 999) <= 50]
    if not num_cols or not cat_cols:
        return None, "No suitable columns for a default chart."
    y_col = num_cols[0]
    x_col = cat_cols[0]
    # Prefer revenue/sales/profit columns
    for keyword in ('profit', 'revenue', 'sales', 'amount', 'price', 'benefit'):
        match = next((c for c in num_cols if keyword in c.lower()), None)
        if match:
            y_col = match
            break
    # Prefer category/product/region columns
    for keyword in ('category', 'product', 'region', 'segment', 'type', 'department'):
        match = next((c for c in cat_cols if keyword in c.lower()), None)
        if match:
            x_col = match
            break
    spec = {
        'chart_type': 'bar', 'x': x_col, 'y': y_col,
        'group_by': x_col, 'aggregation': 'sum',
        'top_n': 10, 'sort': 'desc',
        'title': f'Top 10 {x_col} by {y_col}',
        'x_label': x_col, 'y_label': y_col,
    }
    return execute_chart_spec(df, spec), spec.get('title')


def natural_language_chart(user_query: str, df: pd.DataFrame):
    schema = build_df_schema(df)
    spec = get_chart_spec(user_query, schema)
    if not spec:
        # Fall back to auto-generated default chart
        fig, title = _default_chart(df)
        if fig:
            return fig, f"Auto chart: {title}"
        return None, "Could not understand that request. Try: 'show revenue by category as a bar chart'"
    fig = execute_chart_spec(df, spec)
    if fig is None:
        fig, title = _default_chart(df)
        if fig:
            return fig, f"Auto chart: {title}"
        return None, "Could not build that chart — columns may not contain enough data. Try rephrasing."
    explanation = f"{spec.get('title', 'Chart')} — {spec.get('chart_type', '').title()} chart"
    if spec.get('aggregation') and spec.get('aggregation') != 'none':
        explanation += f" | {spec['aggregation'].title()} of {spec.get('y', '')}"
    if spec.get('group_by'):
        explanation += f" grouped by {spec['group_by']}"
    return fig, explanation