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

def get_customer_metrics(df, cols):
    m = {}
    cust_id = cols.get('customer_id') or cols.get('customer')
    rev_col = cols.get('revenue')
    date_col = cols.get('date')
    qty_col = cols.get('quantity')
    region_col = cols.get('region')
    if not cust_id:
        return m
    m['unique_customers'] = int(df[cust_id].nunique())
    if rev_col:
        clv = df.groupby(cust_id)[rev_col].sum()
        m['avg_clv'] = round(clv.mean(), 2)
        m['median_clv'] = round(clv.median(), 2)
        m['max_clv'] = round(clv.max(), 2)
        sorted_clv = clv.sort_values(ascending=False)
        top20_n = max(1, int(len(sorted_clv) * 0.2))
        m['top20_revenue_share'] = round(sorted_clv.head(top20_n).sum() / sorted_clv.sum() * 100, 1)
        m['top10_customers_revenue'] = round(sorted_clv.head(10).sum(), 2)
    if date_col and rev_col:
        df2 = df[[cust_id, date_col, rev_col]].copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
        df2 = df2.dropna(subset=[date_col])
        snapshot = df2[date_col].max()
        rfm = df2.groupby(cust_id).agg(
            recency=(date_col, lambda x: (snapshot - x.max()).days),
            frequency=(date_col, 'count'),
            monetary=(rev_col, 'sum'),
        ).reset_index()
        try:
            rfm['R'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1], duplicates='drop').astype(float)
            rfm['F'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(float)
            rfm['M'] = pd.qcut(rfm['monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop').astype(float)
            rfm['rfm_score'] = rfm['R'] + rfm['F'] + rfm['M']
            def segment(score):
                if score >= 10: return 'Champions'
                elif score >= 8: return 'Loyal'
                elif score >= 6: return 'Potential'
                elif score >= 4: return 'At Risk'
                else: return 'Lost'
            rfm['segment'] = rfm['rfm_score'].apply(segment)
            m['rfm_segments'] = rfm['segment'].value_counts().to_dict()
            m['champions_pct'] = round((rfm['segment'] == 'Champions').sum() / len(rfm) * 100, 1)
            m['at_risk_pct'] = round((rfm['segment'] == 'At Risk').sum() / len(rfm) * 100, 1)
        except:
            pass
    if region_col and rev_col:
        region_rev = df.groupby(region_col)[rev_col].sum()
        m['top_region'] = str(region_rev.idxmax())
        m['top_region_revenue'] = round(region_rev.max(), 2)
        m['bottom_region'] = str(region_rev.idxmin())
    return m

def plot_rfm_segments(metrics):
    rfm_segs = metrics.get('rfm_segments', {})
    if not rfm_segs:
        return None
    colors = {
        'Champions': '#10b981', 'Loyal': '#38bdf8',
        'Potential': '#f59e0b', 'At Risk': '#f97316', 'Lost': '#ef4444',
    }
    labels = list(rfm_segs.keys())
    values = list(rfm_segs.values())
    chart_colors = [colors.get(l, '#6b7280') for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=chart_colors),
        hole=0.5, textinfo='percent+label',
    ))
    fig.update_layout(title='RFM Customer Segments', **LAYOUT)
    return fig

def plot_clv_distribution(df, cols):
    cust_id = cols.get('customer_id') or cols.get('customer')
    rev_col = cols.get('revenue')
    if not cust_id or not rev_col:
        return None
    clv = df.groupby(cust_id)[rev_col].sum().reset_index()
    clv.columns = ['customer', 'clv']
    fig = go.Figure(go.Histogram(
        x=clv['clv'], nbinsx=40,
        marker_color='#f59e0b', opacity=0.8,
    ))
    fig.update_layout(
        title='Customer Lifetime Value Distribution',
        xaxis_title='Total Revenue per Customer',
        yaxis_title='Customers',
        **LAYOUT
    )
    return fig

def plot_pareto(df, cols):
    cust_id = cols.get('customer_id') or cols.get('customer')
    rev_col = cols.get('revenue')
    if not cust_id or not rev_col:
        return None
    clv = df.groupby(cust_id)[rev_col].sum().sort_values(ascending=False).reset_index()
    clv['cum_pct'] = clv[rev_col].cumsum() / clv[rev_col].sum() * 100
    clv['cust_pct'] = np.arange(1, len(clv) + 1) / len(clv) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clv['cust_pct'], y=clv['cum_pct'],
        name='Revenue Concentration',
        line=dict(color='#f59e0b', width=2),
        fill='tozeroy', fillcolor='rgba(245,158,11,0.08)',
    ))
    fig.add_shape(type='line', x0=0, y0=0, x1=100, y1=100,
        line=dict(color='#1e3a5f', dash='dash'))
    fig.add_shape(type='line', x0=20, y0=0, x1=20, y1=100,
        line=dict(color='#ef4444', dash='dot', width=1))
    fig.update_layout(
        title='Customer Revenue Concentration (Pareto)',
        xaxis_title='% of Customers',
        yaxis_title='% of Revenue',
        **LAYOUT
    )
    return fig

def plot_revenue_by_region(df, cols):
    region_col = cols.get('region')
    rev_col = cols.get('revenue')
    if not region_col or not rev_col:
        return None
    grp = df.groupby(region_col)[rev_col].sum().reset_index()
    grp = grp.sort_values(rev_col, ascending=False).head(12)
    fig = go.Figure(go.Bar(
        x=grp[region_col], y=grp[rev_col],
        marker_color='#38bdf8',
        text=grp[rev_col].apply(lambda x: f'${x:,.0f}'),
        textposition='outside',
    ))
    fig.update_layout(title='Revenue by Region', **LAYOUT)
    return fig