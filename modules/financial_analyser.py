import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LAYOUT = dict(
    paper_bgcolor='#07111f', plot_bgcolor='#07111f',
    font=dict(color='#94a3b8', family='DM Sans', size=12),
    title_font=dict(color='#f1f5f9', family='Syne', size=13),
    xaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f', showgrid=True),
    yaxis=dict(gridcolor='#1e3a5f', zerolinecolor='#1e3a5f', showgrid=True),
    legend=dict(bgcolor='#07111f', bordercolor='#1e3a5f', borderwidth=1),
    margin=dict(t=45, b=35, l=45, r=20),
    height=320,
)

def get_financial_metrics(df, cols):
    rev = cols.get('revenue')
    prof = cols.get('profit')
    cost = cols.get('cost')
    disc = cols.get('discount')
    qty = cols.get('quantity')
    date = cols.get('date')
    cat = cols.get('category')
    m = {}
    if rev:
        m['total_revenue'] = round(df[rev].sum(), 2)
        m['avg_order_value'] = round(df[rev].mean(), 2)
        m['revenue_std'] = round(df[rev].std(), 2)
        m['max_order'] = round(df[rev].max(), 2)
        m['min_order'] = round(df[rev].min(), 2)
    if prof:
        m['total_profit'] = round(df[prof].sum(), 2)
        m['avg_profit'] = round(df[prof].mean(), 2)
        m['profitable_orders_pct'] = round((df[prof] > 0).sum() / len(df) * 100, 1)
    if rev and prof:
        m['profit_margin_pct'] = round(df[prof].sum() / df[rev].sum() * 100, 2) if df[rev].sum() else 0
    if cost and rev:
        m['total_cost'] = round(df[cost].sum(), 2)
        m['cost_ratio_pct'] = round(df[cost].sum() / df[rev].sum() * 100, 2) if df[rev].sum() else 0
    if disc:
        d = df[disc].dropna()
        m['avg_discount_pct'] = round((d.mean() * 100 if d.max() <= 1 else d.mean()), 2)
    if qty:
        m['total_units'] = int(df[qty].sum())
        if rev:
            m['revenue_per_unit'] = round(df[rev].sum() / df[qty].sum(), 2) if df[qty].sum() else 0
    if date and rev:
        df2 = df[[date, rev]].copy()
        df2[date] = pd.to_datetime(df2[date], errors='coerce')
        df2 = df2.dropna()
        monthly = df2.groupby(df2[date].dt.to_period('M'))[rev].sum()
        if len(monthly) >= 2:
            last = monthly.iloc[-1]
            prev = monthly.iloc[-2]
            m['mom_growth_pct'] = round((last - prev) / prev * 100, 1) if prev else 0
    if cat and rev:
        grp = df.groupby(cat)[rev].sum()
        m['best_category'] = str(grp.idxmax())
        m['best_category_revenue'] = round(grp.max(), 2)
        m['worst_category'] = str(grp.idxmin())
        m['worst_category_revenue'] = round(grp.min(), 2)
    return m

def plot_revenue_profit_trend(df, cols):
    date_col = cols.get('date')
    rev_col = cols.get('revenue')
    prof_col = cols.get('profit')
    if not date_col or not rev_col:
        return None
    df2 = df[[date_col, rev_col] + ([prof_col] if prof_col else [])].copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
    df2 = df2.dropna(subset=[date_col])
    df2['month'] = df2[date_col].dt.to_period('M').astype(str)
    agg = {rev_col: 'sum'}
    if prof_col:
        agg[prof_col] = 'sum'
    monthly = df2.groupby('month').agg(agg).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly['month'], y=monthly[rev_col],
        name='Revenue', marker_color='#1e3a5f'), secondary_y=False)
    if prof_col:
        fig.add_trace(go.Scatter(x=monthly['month'], y=monthly[prof_col],
            name='Profit', line=dict(color='#f59e0b', width=2),
            mode='lines+markers', marker=dict(size=3)), secondary_y=True)
    fig.update_layout(title='Revenue & Profit Trend', **LAYOUT)
    fig.update_yaxes(gridcolor='#1e3a5f', secondary_y=False)
    fig.update_yaxes(gridcolor='#1e3a5f', secondary_y=True)
    return fig

def plot_margin_by_category(df, cols):
    cat_col = cols.get('category')
    rev_col = cols.get('revenue')
    prof_col = cols.get('profit')
    if not cat_col or not rev_col or not prof_col:
        return None
    grp = df.groupby(cat_col).agg({rev_col: 'sum', prof_col: 'sum'}).reset_index()
    grp['margin_pct'] = grp.apply(
        lambda r: round(r[prof_col] / r[rev_col] * 100, 1) if r[rev_col] else 0, axis=1
    )
    grp = grp.sort_values('margin_pct', ascending=True).tail(12)
    colors = ['#ef4444' if m < 0 else '#f59e0b' if m < 15 else '#10b981' for m in grp['margin_pct']]
    fig = go.Figure(go.Bar(
        x=grp['margin_pct'], y=grp[cat_col], orientation='h',
        marker_color=colors,
        text=grp['margin_pct'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
    ))
    fig.update_layout(title='Profit Margin by Category', **LAYOUT)
    return fig

def plot_revenue_by_category(df, cols):
    cat_col = cols.get('category')
    rev_col = cols.get('revenue')
    if not cat_col or not rev_col:
        return None
    grp = df.groupby(cat_col)[rev_col].sum().reset_index()
    grp = grp.sort_values(rev_col, ascending=False).head(10)
    fig = go.Figure(go.Bar(
        x=grp[cat_col], y=grp[rev_col], marker_color='#f59e0b',
        text=grp[rev_col].apply(lambda x: f'${x:,.0f}'),
        textposition='outside',
    ))
    fig.update_layout(title='Revenue by Category (Top 10)', **LAYOUT)
    return fig

def plot_discount_impact(df, cols):
    disc_col = cols.get('discount')
    rev_col = cols.get('revenue')
    prof_col = cols.get('profit')
    if not disc_col or not rev_col:
        return None
    df2 = df[[disc_col, rev_col] + ([prof_col] if prof_col else [])].copy().dropna()
    if df2.empty:
        return None
    df2['bucket'] = pd.cut(
        df2[disc_col] * 100 if df2[disc_col].max() <= 1 else df2[disc_col],
        bins=5, precision=0
    ).astype(str)
    agg = {rev_col: 'mean'}
    if prof_col:
        agg[prof_col] = 'mean'
    grp = df2.groupby('bucket').agg(agg).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grp['bucket'], y=grp[rev_col],
        name='Avg Revenue', marker_color='#38bdf8'))
    if prof_col:
        fig.add_trace(go.Scatter(x=grp['bucket'], y=grp[prof_col],
            name='Avg Profit', line=dict(color='#f59e0b', width=2),
            mode='lines+markers'))
    fig.update_layout(title='Discount Rate Impact on Revenue & Profit', **LAYOUT)
    return fig