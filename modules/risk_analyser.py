import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

def get_risk_metrics(df, cols):
    m = {}
    status_col = cols.get('status')
    qty_col = cols.get('quantity')
    rev_col = cols.get('revenue')
    cat_col = cols.get('category')
    region_col = cols.get('region')
    if status_col:
        late = df[status_col].str.lower().str.contains('late', na=False).sum()
        m['late_pct'] = round(late / len(df) * 100, 1)
        m['risk_level'] = (
            'CRITICAL' if m['late_pct'] > 50
            else 'HIGH' if m['late_pct'] > 25
            else 'MEDIUM' if m['late_pct'] > 10
            else 'LOW'
        )
    if qty_col:
        qty = df[qty_col].dropna()
        m['demand_cv'] = round(qty.std() / qty.mean() * 100, 1) if qty.mean() else 0
        m['demand_volatility'] = (
            'HIGH' if m['demand_cv'] > 50
            else 'MEDIUM' if m['demand_cv'] > 25
            else 'LOW'
        )
    if rev_col:
        rev = df[rev_col].dropna()
        m['revenue_at_risk'] = round(rev[rev < 0].sum(), 2) if (rev < 0).any() else 0
        m['var_95'] = round(float(np.percentile(rev, 5)), 2)
    if cat_col and rev_col:
        cat_rev = df.groupby(cat_col)[rev_col].sum()
        top_cat_pct = cat_rev.max() / cat_rev.sum() * 100
        m['top_category_concentration'] = round(top_cat_pct, 1)
        m['concentration_risk'] = (
            'HIGH' if top_cat_pct > 40
            else 'MEDIUM' if top_cat_pct > 25
            else 'LOW'
        )
        m['top_revenue_category'] = str(cat_rev.idxmax())
    if region_col and rev_col:
        reg_rev = df.groupby(region_col)[rev_col].sum()
        m['top_region_concentration'] = round(reg_rev.max() / reg_rev.sum() * 100, 1)
        m['top_revenue_region'] = str(reg_rev.idxmax())
    return m

def plot_risk_matrix(df, cols):
    cat_col = cols.get('category')
    status_col = cols.get('status')
    rev_col = cols.get('revenue')
    if not cat_col or not status_col or not rev_col:
        return None
    df2 = df[[cat_col, status_col, rev_col]].copy()
    df2['is_late'] = df2[status_col].str.lower().str.contains('late', na=False).astype(int)
    grp = df2.groupby(cat_col).agg(
        late_pct=('is_late', lambda x: x.mean() * 100),
        revenue=(rev_col, 'sum'),
        orders=(rev_col, 'count'),
    ).reset_index()
    fig = go.Figure(go.Scatter(
        x=grp['late_pct'],
        y=grp['revenue'],
        mode='markers+text',
        text=grp[cat_col],
        textposition='top center',
        marker=dict(
            size=grp['orders'] / grp['orders'].max() * 35 + 8,
            color=grp['late_pct'],
            colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']],
            showscale=True,
            colorbar=dict(
                title=dict(text='Late %', font=dict(color='#94a3b8')),
                tickfont=dict(color='#94a3b8'),
            ),
        ),
    ))
    fig.add_vline(x=5, line_color='#38bdf8', line_dash='dash',
        annotation_text='Target <5%', annotation_font_color='#38bdf8')
    layout = {**LAYOUT, 'height': 360}
    fig.update_layout(
        title='Risk Matrix: Revenue at Risk vs Late Delivery Rate',
        xaxis_title='Late Delivery Rate (%)',
        yaxis_title='Total Revenue',
        **layout,
    )
    return fig

def plot_demand_volatility(df, cols):
    date_col = cols.get('date')
    qty_col = cols.get('quantity')
    if not date_col or not qty_col:
        return None
    df2 = df[[date_col, qty_col]].copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
    df2 = df2.dropna()
    df2['week'] = df2[date_col].dt.to_period('W').astype(str)
    weekly = df2.groupby('week')[qty_col].sum().reset_index()
    weekly['rolling_avg'] = weekly[qty_col].rolling(4, min_periods=1).mean()
    weekly['upper'] = weekly['rolling_avg'] * 1.2
    weekly['lower'] = weekly['rolling_avg'] * 0.8
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly['week'], y=weekly['upper'],
        fill=None, mode='lines',
        line=dict(color='rgba(239,68,68,0)', width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=weekly['week'], y=weekly['lower'],
        fill='tonexty', mode='lines',
        line=dict(color='rgba(239,68,68,0)', width=0),
        fillcolor='rgba(239,68,68,0.08)', name='+-20% Band',
    ))
    fig.add_trace(go.Scatter(
        x=weekly['week'], y=weekly[qty_col],
        name='Weekly Demand', line=dict(color='#f59e0b', width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=weekly['week'], y=weekly['rolling_avg'],
        name='4-Week Avg', line=dict(color='#38bdf8', width=2, dash='dash'),
    ))
    fig.update_layout(title='Demand Volatility with Risk Bands', **LAYOUT)
    return fig