import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

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

def get_time_intelligence(df, cols):
    date_col = cols.get('date')
    rev_col = cols.get('revenue')
    qty_col = cols.get('quantity')
    target_col = rev_col or qty_col
    if not date_col or not target_col:
        return {}
    df2 = df[[date_col, target_col]].copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
    df2 = df2.dropna(subset=[date_col]).sort_values(date_col)
    latest = df2[date_col].max()
    m = {}
    mtd = df2[(df2[date_col].dt.year == latest.year) & (df2[date_col].dt.month == latest.month)]
    m['mtd'] = round(mtd[target_col].sum(), 2)
    q = (latest.month - 1) // 3
    qtd = df2[(df2[date_col].dt.year == latest.year) & ((df2[date_col].dt.month - 1) // 3 == q)]
    m['qtd'] = round(qtd[target_col].sum(), 2)
    ytd = df2[df2[date_col].dt.year == latest.year]
    m['ytd'] = round(ytd[target_col].sum(), 2)
    m['total'] = round(df2[target_col].sum(), 2)
    this_month = df2[(df2[date_col].dt.year == latest.year) & (df2[date_col].dt.month == latest.month)][target_col].sum()
    prev_month_date = latest - pd.DateOffset(months=1)
    prev_month = df2[(df2[date_col].dt.year == prev_month_date.year) & (df2[date_col].dt.month == prev_month_date.month)][target_col].sum()
    m['mom_pct'] = round((this_month - prev_month) / prev_month * 100, 1) if prev_month else 0
    m['mom_direction'] = 'up' if m['mom_pct'] > 0 else 'down'
    prev_year = df2[df2[date_col].dt.year == latest.year - 1][target_col].sum()
    curr_year = df2[df2[date_col].dt.year == latest.year][target_col].sum()
    m['yoy_pct'] = round((curr_year - prev_year) / prev_year * 100, 1) if prev_year else 0
    m['yoy_direction'] = 'up' if m['yoy_pct'] > 0 else 'down'
    weekly = df2.groupby(df2[date_col].dt.to_period('W'))[target_col].sum()
    m['rolling_4w'] = round(weekly.tail(4).mean(), 2) if len(weekly) >= 4 else round(weekly.mean(), 2)
    m['rolling_12w'] = round(weekly.tail(12).mean(), 2) if len(weekly) >= 12 else round(weekly.mean(), 2)
    monthly = df2.groupby(df2[date_col].dt.to_period('M'))[target_col].sum().reset_index()
    monthly.columns = ['month', 'value']
    monthly['month'] = monthly['month'].astype(str)
    m['monthly_series'] = monthly.to_dict('records')
    return m

def get_variance_analysis(df, cols, targets):
    rev_col = cols.get('revenue')
    qty_col = cols.get('quantity')
    status_col = cols.get('status')
    actuals = {}
    if rev_col:
        actuals['Revenue'] = round(df[rev_col].sum(), 2)
        actuals['Avg Order Value'] = round(df[rev_col].mean(), 2)
    if qty_col:
        actuals['Total Units'] = int(df[qty_col].sum())
    if status_col:
        late = df[status_col].str.lower().str.contains('late', na=False).sum()
        actuals['On-Time Rate (%)'] = round((1 - late / len(df)) * 100, 1)
    results = []
    for metric, actual in actuals.items():
        target = targets.get(metric)
        if target is None:
            continue
        variance = actual - target
        variance_pct = round((variance / target) * 100, 1) if target else 0
        rag = 'GREEN' if variance_pct >= 0 else 'AMBER' if variance_pct >= -10 else 'RED'
        results.append({
            'metric': metric, 'actual': actual, 'target': target,
            'variance': round(variance, 2), 'variance_pct': variance_pct, 'rag': rag,
        })
    return results

def get_rag_scorecard(df, cols, sc_metrics=None, fin_metrics=None):
    scorecard = []
    if fin_metrics:
        margin = fin_metrics.get('profit_margin_pct', 0)
        scorecard.append({
            'kpi': 'Profit Margin', 'value': f"{margin}%", 'benchmark': '>15%',
            'rag': 'GREEN' if margin > 15 else 'AMBER' if margin > 5 else 'RED',
            'comment': 'Healthy' if margin > 15 else 'Below target' if margin > 5 else 'Critical',
        })
        mom = fin_metrics.get('mom_growth_pct', 0)
        scorecard.append({
            'kpi': 'MoM Revenue Growth', 'value': f"{mom:+.1f}%", 'benchmark': '>0%',
            'rag': 'GREEN' if mom > 0 else 'AMBER' if mom > -5 else 'RED',
            'comment': 'Growing' if mom > 0 else 'Declining',
        })
    if sc_metrics:
        otif = sc_metrics.get('otif', 0)
        scorecard.append({
            'kpi': 'OTIF Rate', 'value': f"{otif}%", 'benchmark': '>85%',
            'rag': 'GREEN' if otif > 85 else 'AMBER' if otif > 70 else 'RED',
            'comment': 'On target' if otif > 85 else 'Below benchmark' if otif > 70 else 'Critical',
        })
        late = sc_metrics.get('late_pct', 0)
        scorecard.append({
            'kpi': 'Late Delivery Rate', 'value': f"{late}%", 'benchmark': '<5%',
            'rag': 'GREEN' if late < 5 else 'AMBER' if late < 20 else 'RED',
            'comment': 'Excellent' if late < 5 else 'Needs attention' if late < 20 else 'Critical',
        })
        fill = sc_metrics.get('fill_rate_pct', 0)
        scorecard.append({
            'kpi': 'Fill Rate', 'value': f"{fill}%", 'benchmark': '>95%',
            'rag': 'GREEN' if fill > 95 else 'AMBER' if fill > 85 else 'RED',
            'comment': 'On target' if fill > 95 else 'Needs improvement' if fill > 85 else 'Critical',
        })
    return scorecard

def get_decomposition(df, cols):
    rev_col = cols.get('revenue')
    cat_col = cols.get('category')
    sub_col = cols.get('sub_category')
    prod_col = cols.get('product')
    if not rev_col or not cat_col:
        return {}
    total = df[rev_col].sum()
    result = {'total': round(total, 2), 'categories': []}
    cat_grp = df.groupby(cat_col)[rev_col].sum().sort_values(ascending=False)
    for cat, cat_rev in cat_grp.head(5).items():
        cat_data = {
            'name': cat, 'value': round(cat_rev, 2),
            'pct': round(cat_rev / total * 100, 1), 'children': [],
        }
        cat_df = df[df[cat_col] == cat]
        drill_col = sub_col or prod_col
        if drill_col:
            drill_grp = cat_df.groupby(drill_col)[rev_col].sum().sort_values(ascending=False)
            for drill, drill_rev in drill_grp.head(3).items():
                cat_data['children'].append({
                    'name': str(drill)[:30], 'value': round(drill_rev, 2),
                    'pct': round(drill_rev / cat_rev * 100, 1), 'children': [],
                })
        result['categories'].append(cat_data)
    return result

def get_sensitivity_analysis(df, cols):
    rev_col = cols.get('revenue')
    if not rev_col:
        return {}
    numeric = df.select_dtypes(include=[np.number])
    if rev_col not in numeric.columns:
        return {}
    target = numeric[rev_col].dropna()
    results = []
    for col in numeric.columns:
        if col == rev_col:
            continue
        series = numeric[col].dropna()
        common_idx = target.index.intersection(series.index)
        if len(common_idx) < 10:
            continue
        try:
            corr, pval = stats.pearsonr(target.loc[common_idx], series.loc[common_idx])
            results.append({
                'variable': str(col), 'correlation': float(round(corr, 3)),
                'abs_correlation': float(abs(round(corr, 3))),
                'p_value': float(round(pval, 4)), 'significant': bool(pval < 0.05),
                'impact': 'HIGH' if abs(corr) > 0.5 else 'MEDIUM' if abs(corr) > 0.25 else 'LOW',
                'direction': 'Positive' if corr > 0 else 'Negative',
            })
        except:
            continue
    results.sort(key=lambda x: x['abs_correlation'], reverse=True)
    return {'drivers': results[:10], 'target': rev_col}

def get_pivot_summary(df, cols):
    rev_col = cols.get('revenue')
    qty_col = cols.get('quantity')
    cat_col = cols.get('category')
    region_col = cols.get('region')
    results = {}
    if cat_col and rev_col:
        if region_col:
            try:
                pivot = pd.pivot_table(
                    df, values=rev_col, index=cat_col,
                    columns=region_col, aggfunc='sum', fill_value=0,
                ).round(0)
                pivot['TOTAL'] = pivot.sum(axis=1)
                results['category_region_revenue'] = pivot.head(10)
            except:
                pass
        cat_summary = df.groupby(cat_col).agg(
            Revenue=(rev_col, 'sum'),
            Orders=(rev_col, 'count'),
            Avg_Order=(rev_col, 'mean'),
        ).round(2).sort_values('Revenue', ascending=False).head(15)
        if qty_col:
            cat_summary['Units'] = df.groupby(cat_col)[qty_col].sum()
        results['category_summary'] = cat_summary
    return results

def get_funnel_analysis(df, cols):
    status_col = cols.get('status')
    if not status_col:
        return {}
    total = len(df)
    vc = df[status_col].str.lower()
    stages = {
        'Orders Placed': total,
        'Not Cancelled': int((~vc.str.contains('cancel', na=False)).sum()),
        'Shipped': int((vc.str.contains('ship|advance|time|late', na=False)).sum()),
        'On-Time / Early': int((vc.str.contains('time|advance', na=False)).sum()),
    }
    results = []
    prev = total
    for stage, count in stages.items():
        conv = round(count / total * 100, 1)
        drop = round((prev - count) / prev * 100, 1) if prev > 0 and stage != 'Orders Placed' else 0
        results.append({'stage': stage, 'count': count, 'conversion_pct': conv, 'drop_pct': drop})
        prev = count
    return {'funnel': results}

def plot_waterfall(df, cols):
    cat_col = cols.get('category')
    rev_col = cols.get('revenue')
    cost_col = cols.get('cost')
    if not cat_col or not rev_col:
        return None
    grp = df.groupby(cat_col)[rev_col].sum().sort_values(ascending=False).head(6)
    labels = ['Total'] + list(grp.index) + (['Cost Deduction'] if cost_col else [])
    total = grp.sum()
    values = [total] + list(grp.values)
    measures = ['total'] + ['relative'] * len(grp)
    if cost_col:
        values.append(-df[cost_col].sum())
        measures.append('relative')
    fig = go.Figure(go.Waterfall(
        name='Revenue', orientation='v',
        measure=measures, x=labels, y=values,
        connector=dict(line=dict(color='#1e3a5f', width=1)),
        increasing=dict(marker=dict(color='#10b981')),
        decreasing=dict(marker=dict(color='#ef4444')),
        totals=dict(marker=dict(color='#f59e0b')),
        textposition='outside',
        text=[f'${v:,.0f}' for v in values],
    ))
    fig.update_layout(title='Revenue Waterfall by Category', **LAYOUT)
    return fig

def plot_time_intelligence(time_metrics):
    if not time_metrics or 'monthly_series' not in time_metrics:
        return None
    monthly = pd.DataFrame(time_metrics['monthly_series'])
    if monthly.empty:
        return None
    monthly['rolling_3'] = monthly['value'].rolling(3, min_periods=1).mean()
    monthly['rolling_6'] = monthly['value'].rolling(6, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly['month'], y=monthly['value'],
        name='Monthly', marker_color='#1e3a5f'))
    fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['rolling_3'],
        name='3M Avg', line=dict(color='#f59e0b', width=2)))
    fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['rolling_6'],
        name='6M Avg', line=dict(color='#38bdf8', width=2, dash='dash')))
    fig.update_layout(title='Monthly Performance with Rolling Averages', **LAYOUT)
    return fig

def plot_variance_chart(variance_results):
    if not variance_results:
        return None
    rag_colors = {'GREEN': '#10b981', 'AMBER': '#f59e0b', 'RED': '#ef4444'}
    metrics = [v['metric'] for v in variance_results]
    variances = [v['variance_pct'] for v in variance_results]
    colors = [rag_colors[v['rag']] for v in variance_results]
    fig = go.Figure(go.Bar(
        x=metrics, y=variances, marker_color=colors,
        text=[f'{v:+.1f}%' for v in variances], textposition='outside',
    ))
    fig.add_hline(y=0, line_color='#4a6580', line_dash='dash')
    fig.update_layout(title='Variance vs Target (%)', yaxis_title='Variance %', **LAYOUT)
    return fig

def plot_sensitivity(sensitivity):
    if not sensitivity or not sensitivity.get('drivers'):
        return None
    drivers = sensitivity['drivers']
    variables = [d['variable'][:25] for d in drivers]
    correlations = [d['correlation'] for d in drivers]
    colors = ['#10b981' if c > 0 else '#ef4444' for c in correlations]
    significance = ['* ' if d['significant'] else '' for d in drivers]
    labels = [f"{s}{v}" for s, v in zip(significance, variables)]
    fig = go.Figure(go.Bar(
        x=correlations, y=labels, orientation='h',
        marker_color=colors,
        text=[f'{c:+.3f}' for c in correlations],
        textposition='outside',
    ))
    fig.add_vline(x=0, line_color='#4a6580')
    fig.add_vline(x=0.5, line_color='#10b981', line_dash='dot')
    fig.add_vline(x=-0.5, line_color='#ef4444', line_dash='dot')
    fig.update_layout(
        title=f'Sensitivity Analysis - Drivers of {sensitivity["target"]} (* = significant)',
        xaxis_title='Pearson Correlation',
        **{**LAYOUT, 'height': 360},
    )
    return fig

def plot_funnel(funnel_data):
    if not funnel_data or not funnel_data.get('funnel'):
        return None
    funnel = funnel_data['funnel']
    fig = go.Figure(go.Funnel(
        y=[f['stage'] for f in funnel],
        x=[f['count'] for f in funnel],
        textinfo='value+percent initial',
        marker=dict(color=['#f59e0b', '#38bdf8', '#10b981', '#a78bfa']),
        connector=dict(line=dict(color='#1e3a5f', width=1)),
    ))
    fig.update_layout(
        title='Order Fulfillment Funnel',
        paper_bgcolor='#07111f', plot_bgcolor='#07111f',
        font=dict(color='#94a3b8', family='DM Sans'),
        title_font=dict(color='#f1f5f9', family='Syne', size=13),
        margin=dict(t=45, b=35, l=120, r=20), height=300,
    )
    return fig

def plot_decomposition_treemap(decomp):
    if not decomp or not decomp.get('categories'):
        return None
    labels = ['Total']
    parents = ['']
    values = [decomp['total']]
    for cat in decomp['categories']:
        labels.append(cat['name'])
        parents.append('Total')
        values.append(cat['value'])
        for child in cat.get('children', []):
            labels.append(child['name'])
            parents.append(cat['name'])
            values.append(child['value'])
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        textinfo='label+percent parent',
        marker=dict(
            colorscale=[[0, '#0a1628'], [0.5, '#1e3a5f'], [1, '#f59e0b']],
            line=dict(color='#07111f', width=2),
        ),
    ))
    fig.update_layout(
        title='Revenue Decomposition Tree',
        paper_bgcolor='#07111f',
        font=dict(color='#f1f5f9', family='DM Sans'),
        title_font=dict(color='#f1f5f9', family='Syne', size=13),
        margin=dict(t=45, b=10, l=10, r=10), height=380,
    )
    return fig