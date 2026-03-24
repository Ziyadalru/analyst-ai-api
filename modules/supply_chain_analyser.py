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

BENCHMARKS = {'otif': 85.0, 'late_pct': 5.0, 'cycle_days': 3.0, 'fill_rate': 95.0}

def get_supply_chain_metrics(df, cols):
    m = {}
    status_col = cols.get('status')
    date_col = cols.get('date')
    ship_col = cols.get('ship_date')
    days_real = cols.get('days_real')
    days_sched = cols.get('days_scheduled')
    qty_col = cols.get('quantity')
    supplier_col = cols.get('supplier')
    m['total_orders'] = len(df)
    if status_col:
        total = len(df)
        late = df[status_col].str.lower().str.contains('late', na=False).sum()
        on_time = df[status_col].str.lower().str.contains('time|advance', na=False).sum()
        cancelled = df[status_col].str.lower().str.contains('cancel', na=False).sum()
        m['late_orders'] = int(late)
        m['late_pct'] = round(late / total * 100, 1)
        m['on_time_pct'] = round(on_time / total * 100, 1)
        m['cancelled_orders'] = int(cancelled)
        m['cancelled_pct'] = round(cancelled / total * 100, 1)
        m['otif'] = round(on_time / total * 100, 1)
        m['otif_benchmark'] = BENCHMARKS['otif']
        m['otif_gap'] = round(m['otif'] - BENCHMARKS['otif'], 1)
        m['status_breakdown'] = df[status_col].value_counts().to_dict()
        filled = total - int(cancelled)
        m['fill_rate_pct'] = round(filled / total * 100, 1)
        m['fill_rate_benchmark'] = BENCHMARKS['fill_rate']
    if date_col and ship_col:
        df2 = df[[date_col, ship_col]].copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
        df2[ship_col] = pd.to_datetime(df2[ship_col], errors='coerce')
        df2['cycle'] = (df2[ship_col] - df2[date_col]).dt.days
        df2 = df2[(df2['cycle'] >= 0) & (df2['cycle'] < 365)]
        if not df2.empty:
            m['avg_cycle_days'] = round(df2['cycle'].mean(), 1)
            m['median_cycle_days'] = round(df2['cycle'].median(), 1)
            m['cycle_std'] = round(df2['cycle'].std(), 1)
    if days_real and days_sched:
        df3 = df[[days_real, days_sched]].dropna()
        df3 = df3[(df3[days_real] >= 0) & (df3[days_sched] >= 0)]
        df3['delay'] = df3[days_real] - df3[days_sched]
        m['avg_delay_days'] = round(df3['delay'].mean(), 1)
        m['pct_delayed'] = round((df3['delay'] > 0).sum() / len(df3) * 100, 1)
    if qty_col:
        qty = df[qty_col].dropna()
        m['avg_order_qty'] = round(qty.mean(), 1)
        m['qty_cv'] = round(qty.std() / qty.mean() * 100, 1) if qty.mean() else 0
        if days_real:
            avg_demand = qty.mean()
            demand_std = qty.std()
            lead_time = df[days_real].mean()
            lead_std = df[days_real].std()
            z = 1.65
            safety_stock = z * np.sqrt(
                (lead_time * demand_std**2) + (avg_demand**2 * lead_std**2)
            )
            m['safety_stock_estimate'] = round(safety_stock, 0)
            m['reorder_point'] = round(avg_demand * lead_time + safety_stock, 0)
    if supplier_col and status_col:
        try:
            sup_late = df.groupby(supplier_col).apply(
                lambda x: x[status_col].str.lower().str.contains('late', na=False).mean() * 100
            ).round(1)
            m['worst_supplier'] = str(sup_late.idxmax())
            m['worst_supplier_late_pct'] = round(sup_late.max(), 1)
            m['best_supplier'] = str(sup_late.idxmin())
            m['best_supplier_late_pct'] = round(sup_late.min(), 1)
        except:
            pass
    return m

def plot_otif_gauge(metrics):
    otif = metrics.get('otif', 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=otif,
        delta={'reference': BENCHMARKS['otif'], 'valueformat': '.1f'},
        title={'text': "OTIF Rate (%)", 'font': {'color': '#f1f5f9', 'family': 'Syne', 'size': 13}},
        number={'font': {'color': '#f59e0b', 'family': 'IBM Plex Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#4a6580'},
            'bar': {'color': '#f59e0b'},
            'bgcolor': '#0a1628',
            'bordercolor': '#1e3a5f',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239,68,68,0.2)'},
                {'range': [50, 85], 'color': 'rgba(245,158,11,0.2)'},
                {'range': [85, 100], 'color': 'rgba(16,185,129,0.2)'},
            ],
            'threshold': {
                'line': {'color': '#38bdf8', 'width': 2},
                'thickness': 0.75,
                'value': BENCHMARKS['otif'],
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor='#07111f',
        font=dict(color='#94a3b8'),
        margin=dict(t=60, b=20, l=20, r=20),
        height=280,
    )
    return fig

def plot_delivery_status(df, cols):
    status_col = cols.get('status')
    if not status_col:
        return None
    vc = df[status_col].value_counts().reset_index()
    vc.columns = ['status', 'count']
    color_map = {
        'Late delivery': '#ef4444',
        'Advance shipping': '#10b981',
        'Shipping on time': '#38bdf8',
        'Shipping canceled': '#6b7280',
    }
    colors = [color_map.get(s, '#f59e0b') for s in vc['status']]
    fig = go.Figure(go.Pie(
        labels=vc['status'], values=vc['count'],
        marker=dict(colors=colors),
        hole=0.5, textinfo='percent+label', textfont=dict(size=11),
    ))
    fig.update_layout(title='Delivery Status Breakdown', **LAYOUT)
    return fig

def plot_late_by_category(df, cols):
    cat_col = cols.get('category')
    status_col = cols.get('status')
    if not cat_col or not status_col:
        return None
    df2 = df[[cat_col, status_col]].copy()
    df2['is_late'] = df2[status_col].str.lower().str.contains('late', na=False).astype(int)
    grp = df2.groupby(cat_col)['is_late'].agg(['sum', 'count']).reset_index()
    grp['late_pct'] = (grp['sum'] / grp['count'] * 100).round(1)
    grp = grp.sort_values('late_pct', ascending=True).tail(12)
    colors = ['#ef4444' if x > 60 else '#f59e0b' if x > 30 else '#10b981' for x in grp['late_pct']]
    fig = go.Figure(go.Bar(
        x=grp['late_pct'], y=grp[cat_col], orientation='h',
        marker_color=colors,
        text=grp['late_pct'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
    ))
    fig.add_vline(x=BENCHMARKS['late_pct'], line_color='#38bdf8', line_dash='dash',
        annotation_text=f'Benchmark {BENCHMARKS["late_pct"]}%',
        annotation_font_color='#38bdf8')
    fig.update_layout(title='Late Delivery Rate by Category vs Benchmark', **LAYOUT)
    return fig

def plot_cycle_time_trend(df, cols):
    date_col = cols.get('date')
    ship_col = cols.get('ship_date')
    if not date_col or not ship_col:
        return None
    df2 = df[[date_col, ship_col]].copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
    df2[ship_col] = pd.to_datetime(df2[ship_col], errors='coerce')
    df2['cycle'] = (df2[ship_col] - df2[date_col]).dt.days
    df2 = df2[(df2['cycle'] >= 0) & (df2['cycle'] < 365)]
    if df2.empty:
        return None
    df2['month'] = df2[date_col].dt.to_period('M').astype(str)
    monthly = df2.groupby('month')['cycle'].agg(['mean', 'median']).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['mean'],
        name='Avg Cycle Time', line=dict(color='#f59e0b', width=2), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['median'],
        name='Median', line=dict(color='#38bdf8', width=2, dash='dash')))
    fig.update_layout(title='Order Fulfillment Cycle Time (Days)', **LAYOUT)
    return fig

def plot_supplier_scorecard(df, cols):
    supplier_col = cols.get('supplier')
    status_col = cols.get('status')
    if not supplier_col or not status_col:
        return None
    df2 = df[[supplier_col, status_col]].copy()
    df2['is_late'] = df2[status_col].str.lower().str.contains('late', na=False).astype(int)
    grp = df2.groupby(supplier_col)['is_late'].agg(['mean', 'count']).reset_index()
    grp.columns = [supplier_col, 'late_rate', 'order_count']
    grp['late_rate'] = (grp['late_rate'] * 100).round(1)
    grp['on_time_rate'] = (100 - grp['late_rate']).round(1)
    grp = grp.sort_values('on_time_rate', ascending=True).tail(10)
    colors = ['#ef4444' if x < 50 else '#f59e0b' if x < 85 else '#10b981' for x in grp['on_time_rate']]
    fig = go.Figure(go.Bar(
        x=grp['on_time_rate'], y=grp[supplier_col], orientation='h',
        marker_color=colors,
        text=grp['on_time_rate'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
    ))
    fig.add_vline(x=85, line_color='#38bdf8', line_dash='dash',
        annotation_text='Benchmark 85%', annotation_font_color='#38bdf8')
    fig.update_layout(title='Supplier On-Time Rate Scorecard', **LAYOUT)
    return fig