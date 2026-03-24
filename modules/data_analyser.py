import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from groq import Groq

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

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


# ── AI insights ───────────────────────────────────────────────────────────────
def _ai_data_insights(df: pd.DataFrame, cols: dict) -> list[str]:
    """Ask Groq for 4 specific, data-driven findings about this dataset."""
    try:
        numeric = df.select_dtypes(include=[np.number])
        describe_str = numeric.describe().round(2).to_string()[:1200] if not numeric.empty else "No numeric columns"

        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        cat_summary = {}
        for col in cat_cols[:5]:
            vc = df[col].value_counts()
            cat_summary[col] = {
                'unique': int(df[col].nunique()),
                'top': str(vc.index[0]) if len(vc) else None,
                'top_pct': round(vc.iloc[0] / len(df) * 100, 1) if len(vc) else 0,
            }

        active_cols = {k: v for k, v in cols.items() if v}

        prompt = f"""You are a data analyst. Analyse this dataset and give 4 specific, interesting findings.

Dataset: {len(df)} rows, {len(df.columns)} columns
Numeric column statistics:
{describe_str}

Categorical columns: {json.dumps(cat_summary)}
Detected column roles: {json.dumps(active_cols)}

Give exactly 4 findings. Each must:
- Use actual numbers from the data
- Explain what it means or why it matters
- Be one sentence

Return ONLY a JSON array, no explanation:
["finding 1", "finding 2", "finding 3", "finding 4"]"""

        res = _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
            timeout=8.0,
        )
        raw = res.choices[0].message.content.strip()
        start, end = raw.find('['), raw.rfind(']')
        if start == -1 or end == -1:
            return []
        return json.loads(raw[start:end + 1])
    except Exception:
        return []


# ── Helpers for dynamic chart selection ──────────────────────────────────────
def _most_interesting_numeric(df: pd.DataFrame, cols: list[str]) -> str:
    """Column with highest coefficient of variation — most spread relative to mean."""
    best, best_cv = cols[0], 0.0
    for col in cols:
        data = df[col].dropna()
        if len(data) < 5 or data.mean() == 0:
            continue
        cv = data.std() / abs(data.mean())
        if cv > best_cv:
            best_cv, best = cv, col
    return best


def _best_categorical(df: pd.DataFrame, cat_cols: list[str]) -> str:
    """Prefer a categorical col with 2–20 unique values — best for grouping."""
    for col in cat_cols:
        n = df[col].nunique()
        if 2 <= n <= 20:
            return col
    return cat_cols[0]


def _top_correlated_pair(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[str | None, str | None]:
    """Find the two numeric columns with the highest absolute correlation."""
    if len(numeric_cols) < 2:
        return None, None
    numeric = df[numeric_cols].select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None, None
    corr = numeric.corr().abs()
    np.fill_diagonal(corr.values, 0)
    stacked = corr.stack()
    if stacked.empty:
        return None, None
    best = stacked.idxmax()
    return best[0], best[1]


# ── Metrics ───────────────────────────────────────────────────────────────────
def get_data_insights(df: pd.DataFrame, cols: dict) -> dict:
    """Dynamic data analysis — AI-generated insights + key statistics."""
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(include=['object'])

    m: dict = {
        'numeric_cols': list(numeric.columns),
        'categorical_cols': list(categorical.columns),
    }

    if not numeric.empty:
        # Outliers via z-score
        outlier_counts: dict[str, int] = {}
        for col in numeric.columns:
            z = np.abs(stats.zscore(numeric[col].dropna()))
            outlier_counts[col] = int((z > 3).sum())
        m['outliers'] = outlier_counts
        m['total_outliers'] = sum(outlier_counts.values())
        m['most_outliers_col'] = max(outlier_counts, key=outlier_counts.get) if outlier_counts else None
        m['skewness'] = {k: (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else round(v, 2))
                         for k, v in numeric.skew().items()}

        # Top correlated pairs
        if numeric.shape[1] >= 2:
            corr = numeric.corr()
            pairs = []
            cols_list = list(corr.columns)
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    val = corr.iloc[i, j]
                    if not np.isnan(val):
                        pairs.append({'col_a': cols_list[i], 'col_b': cols_list[j], 'r': round(float(val), 2)})
            pairs.sort(key=lambda x: abs(x['r']), reverse=True)
            m['top_correlations'] = pairs[:5]

    # AI-generated insights (the main value-add)
    m['ai_insights'] = _ai_data_insights(df, cols)

    return m


# ── Charts ────────────────────────────────────────────────────────────────────
def plot_correlation_matrix(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None
    cols = numeric.columns[:12]
    corr = numeric[cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0, '#ef4444'], [0.5, '#07111f'], [1, '#10b981']],
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont=dict(size=9),
        colorbar=dict(
            title=dict(text='Correlation', font=dict(color='#94a3b8')),
            tickfont=dict(color='#94a3b8'),
        ),
    ))
    fig.update_layout(title='Correlation Matrix', **{**LAYOUT, 'height': 380})
    return fig


def plot_distribution(df: pd.DataFrame, col: str):
    data = df[col].dropna()
    if len(data) < 10:
        return None
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data, nbinsx=40,
        marker_color='#f59e0b', opacity=0.7, name='Distribution',
    ))
    mean, std = data.mean(), data.std()
    x_range = np.linspace(data.min(), data.max(), 100)
    normal_curve = stats.norm.pdf(x_range, mean, std) * len(data) * (data.max() - data.min()) / 40
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_curve,
        line=dict(color='#38bdf8', width=2), name='Normal Curve',
    ))
    fig.update_layout(title=f'Distribution: {col}', **LAYOUT)
    return fig


def plot_outliers(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).iloc[:, :6]
    if numeric.empty:
        return None
    fig = go.Figure()
    for col in numeric.columns:
        fig.add_trace(go.Box(
            y=numeric[col].dropna(), name=col,
            marker_color='#f59e0b',
            line_color='#f59e0b',
            fillcolor='rgba(245,158,11,0.1)',
        ))
    fig.update_layout(title='Outlier Detection (Box Plot)', **{**LAYOUT, 'height': 340})
    return fig


def plot_category_breakdown(df: pd.DataFrame, cat_col: str, num_col: str):
    """Average of a numeric metric grouped by a categorical column."""
    try:
        grp = (
            df.groupby(cat_col)[num_col]
            .mean()
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
        )
        if grp.empty:
            return None
        fig = go.Figure(go.Bar(
            x=grp[cat_col].astype(str),
            y=grp[num_col],
            marker_color='#f59e0b',
            marker_line_color='#07111f',
            marker_line_width=1,
        ))
        fig.update_layout(title=f'Average {num_col} by {cat_col}', **LAYOUT)
        return fig
    except Exception:
        return None


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    """Scatter plot of two numeric columns."""
    try:
        data = df[[x_col, y_col]].dropna()
        if len(data) < 5:
            return None
        sample = data.sample(min(500, len(data)), random_state=42)
        fig = go.Figure(go.Scatter(
            x=sample[x_col], y=sample[y_col],
            mode='markers',
            marker=dict(color='#38bdf8', size=5, opacity=0.6),
        ))
        fig.update_layout(
            title=f'{x_col} vs {y_col}',
            xaxis=dict(**LAYOUT['xaxis'], title=x_col),
            yaxis=dict(**LAYOUT['yaxis'], title=y_col),
            **{k: v for k, v in LAYOUT.items() if k not in ('xaxis', 'yaxis')},
        )
        return fig
    except Exception:
        return None


def plot_abc_analysis(df: pd.DataFrame, cols: dict):
    prod_col = cols.get('product') or cols.get('category')
    rev_col = cols.get('revenue') or cols.get('quantity')
    if not prod_col or not rev_col:
        return None
    grp = df.groupby(prod_col)[rev_col].sum().sort_values(ascending=False).reset_index()
    grp['cum_pct'] = grp[rev_col].cumsum() / grp[rev_col].sum() * 100
    grp['class'] = grp['cum_pct'].apply(lambda x: 'A' if x <= 70 else 'B' if x <= 90 else 'C')
    color_map = {'A': '#10b981', 'B': '#f59e0b', 'C': '#ef4444'}
    fig = go.Figure()
    for cls in ['A', 'B', 'C']:
        sub = grp[grp['class'] == cls]
        fig.add_trace(go.Bar(
            x=sub[prod_col].astype(str).str[:20], y=sub[rev_col],
            name=f'Class {cls}', marker_color=color_map[cls],
        ))
    fig.update_layout(
        title='ABC Analysis (A=Top 70%, B=Next 20%, C=Bottom 10%)',
        barmode='stack', **{**LAYOUT, 'height': 340},
    )
    return fig


# ── Dynamic chart selector ────────────────────────────────────────────────────
def get_dynamic_charts(df: pd.DataFrame, cols: dict) -> dict:
    """Pick the most relevant charts for this specific dataset."""
    charts: dict = {}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object']).columns.tolist()

    # Correlation matrix (if 2+ numeric cols)
    if len(numeric) >= 2:
        charts['correlation_matrix'] = plot_correlation_matrix(df)

    # Distribution of most interesting numeric column
    if numeric:
        best_col = _most_interesting_numeric(df, numeric)
        charts['distribution'] = plot_distribution(df, best_col)

    # Outlier box plot
    if numeric:
        charts['outliers'] = plot_outliers(df)

    # Category breakdown — best categorical vs best numeric
    if categorical and numeric:
        cat_col = _best_categorical(df, categorical)
        num_col = numeric[0]
        charts['category_breakdown'] = plot_category_breakdown(df, cat_col, num_col)

    # Scatter of most correlated pair
    if len(numeric) >= 2:
        x_col, y_col = _top_correlated_pair(df, numeric)
        if x_col and y_col and x_col != y_col:
            charts['scatter'] = plot_scatter(df, x_col, y_col)

    # ABC analysis if product + revenue cols detected
    if cols.get('product') or cols.get('category'):
        if cols.get('revenue') or cols.get('quantity'):
            charts['abc_analysis'] = plot_abc_analysis(df, cols)

    return charts


# ── Kept for backward compat (analyse.py imports this) ───────────────────────
def get_statistical_metrics(df: pd.DataFrame) -> dict:
    return get_data_insights(df, {})
