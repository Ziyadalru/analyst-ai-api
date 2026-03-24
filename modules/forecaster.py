import pandas as pd
import numpy as np
from prophet import Prophet
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

def prepare_forecast_data(df, cols):
    date_col = cols.get('date')
    qty_col = cols.get('quantity')
    rev_col = cols.get('revenue')
    target_col = qty_col or rev_col
    if not date_col or not target_col:
        return None
    df2 = df[[date_col, target_col]].copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
    df2 = df2.dropna()
    df2['week'] = df2[date_col].dt.to_period('W').dt.start_time
    weekly = df2.groupby('week')[target_col].sum().reset_index()
    weekly.columns = ['ds', 'y']
    weekly = weekly[weekly['y'] >= 0]
    return weekly

def run_forecast(demand_df, periods=13):
    if demand_df is None or len(demand_df) < 4:
        return None, None
    try:
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=len(demand_df) > 52,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            seasonality_mode='additive',
        )
        model.fit(demand_df)
        future = model.make_future_dataframe(periods=periods, freq='W')
        forecast = model.predict(future)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        return model, forecast
    except Exception:
        return None, None

def get_forecast_metrics(forecast, demand_df):
    if forecast is None or demand_df is None:
        return {}
    future_fc = forecast[forecast['ds'] > demand_df['ds'].max()]
    hist_avg = demand_df['y'].mean()
    avg = future_fc['yhat'].mean() if not future_fc.empty else hist_avg
    peak = future_fc['yhat'].max() if not future_fc.empty else hist_avg
    peak_date = (
        future_fc.loc[future_fc['yhat'].idxmax(), 'ds']
        if not future_fc.empty
        else demand_df['ds'].max()
    )
    if len(demand_df) >= 4:
        recent = demand_df['y'].tail(8).mean()
        older = demand_df['y'].head(8).mean()
        trend_pct = round((recent - older) / older * 100, 1) if older else 0
    else:
        trend_pct = 0
    return {
        'avg_forecast': round(avg, 1),
        'peak_forecast': round(peak, 1),
        'peak_date': peak_date.strftime('%Y-%m-%d') if hasattr(peak_date, 'strftime') else str(peak_date),
        'hist_avg': round(hist_avg, 1),
        'trend_pct': trend_pct,
        'trend_direction': 'Growing' if trend_pct > 2 else 'Declining' if trend_pct < -2 else 'Stable',
    }

def plot_forecast_chart(forecast, demand_df):
    if forecast is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        fill=None, mode='lines',
        line=dict(color='rgba(245,158,11,0)', width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        fill='tonexty', mode='lines',
        line=dict(color='rgba(245,158,11,0)', width=0),
        fillcolor='rgba(245,158,11,0.1)', name='Confidence Interval',
    ))
    fig.add_trace(go.Scatter(
        x=demand_df['ds'], y=demand_df['y'],
        name='Historical', line=dict(color='#38bdf8', width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        name='Forecast', line=dict(color='#f59e0b', width=2, dash='dash'),
    ))
    if demand_df is not None:
        cutoff = demand_df['ds'].max()
        fig.add_vline(
            x=cutoff.timestamp() * 1000,
            line_color='#4a6580', line_dash='dot',
            annotation_text='Forecast Start',
            annotation_font_color='#4a6580',
        )
    fig.update_layout(title='90-Day Demand Forecast', **LAYOUT)
    return fig