def detect_anomalies(quality, fin_m, sc_m, cust_m, risk_m, fc_m, exec_m) -> list[dict]:
    """Return list of {severity, message} dicts. severity: critical|warning|info."""
    alerts = []

    def a(severity, msg):
        alerts.append({"severity": severity, "message": msg})

    qs = quality.get("quality_score", 100)
    if qs < 60:
        a("critical", f"Data quality score is {qs}/100 — analyses may be unreliable. Clean missing/duplicate records before acting on results.")
    elif qs < 80:
        a("warning", f"Data quality score is {qs}/100 — some metrics may be skewed by missing or duplicate data.")

    if fin_m:
        margin = fin_m.get("profit_margin_pct")
        if margin is not None and margin < 5:
            a("critical", f"Profit margin is {margin}% — critically below the 10% minimum healthy threshold. Immediate cost or pricing review required.")
        elif margin is not None and margin < 10:
            a("warning", f"Profit margin is {margin}% — below the 10–20% healthy range. Monitor cost structure closely.")

        mom = fin_m.get("mom_growth_pct")
        if mom is not None and mom < -15:
            a("critical", f"Revenue declined {abs(mom):.1f}% month-over-month — investigate root cause immediately.")
        elif mom is not None and mom < -5:
            a("warning", f"Revenue declined {abs(mom):.1f}% month-over-month — monitor trend over next 2 periods.")

        discount = fin_m.get("avg_discount_pct")
        if discount is not None and discount > 25:
            a("warning", f"Average discount is {discount}% — excessive discounting is eroding margins. Review pricing strategy.")

    if sc_m:
        otif = sc_m.get("otif")
        if otif is not None and otif < 75:
            a("critical", f"OTIF rate is {otif}% — far below the 95% SCOR benchmark. Customer satisfaction is at serious risk.")
        elif otif is not None and otif < 85:
            a("warning", f"OTIF rate is {otif}% — below the 85% minimum threshold. Delivery performance needs attention.")

        late = sc_m.get("late_pct")
        if late is not None and late > 20:
            a("critical", f"{late}% of deliveries are late — well above the 5% industry benchmark.")
        elif late is not None and late > 10:
            a("warning", f"{late}% of deliveries are late — above the 5% best-practice threshold.")

    if cust_m:
        top20 = cust_m.get("top20_revenue_share")
        if top20 is not None and top20 > 80:
            a("critical", f"Top 20% of customers drive {top20}% of revenue — extreme concentration risk.")
        elif top20 is not None and top20 > 70:
            a("warning", f"Top 20% of customers drive {top20}% of revenue — high concentration. Diversify customer base.")

        champs = cust_m.get("champions_pct")
        if champs is not None and champs < 5:
            a("warning", f"Only {champs}% of customers are Champions — loyalty program or re-engagement campaign recommended.")

    if risk_m:
        level = risk_m.get("risk_level", "")
        if level == "CRITICAL":
            a("critical", "Overall risk level is CRITICAL — multiple risk thresholds breached simultaneously.")
        elif level == "HIGH":
            a("warning", "Overall risk level is HIGH — proactive mitigation steps should be taken within 30 days.")

        cv = risk_m.get("demand_cv")
        if cv is not None and cv > 50:
            a("warning", f"Demand volatility (CV) is {cv}% — high unpredictability increases inventory costs and stockout risk.")

        conc = risk_m.get("top_category_concentration")
        if conc is not None and conc > 60:
            a("warning", f"Top category drives {conc}% of volume — category concentration risk.")

    if fc_m:
        trend_pct = fc_m.get("trend_pct")
        if trend_pct is not None and trend_pct < -10:
            a("critical", f"Demand forecast shows a {abs(trend_pct):.1f}% declining trend — review product strategy.")

    if exec_m and exec_m.get("time"):
        yoy = exec_m["time"].get("yoy_pct")
        if yoy is not None and yoy < -10:
            a("critical", f"Year-over-year revenue is down {abs(yoy):.1f}% — significant annual performance deterioration.")

    return alerts
