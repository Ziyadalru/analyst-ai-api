import os
import tempfile
from fpdf import FPDF
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Text colours for white paper ─────────────────────────────────────────────
C_TITLE   = (10,  40,  80)   # deep navy      — main headings
C_ACCENT  = (210,  80,   0)  # vivid orange   — section labels / bullets
C_BODY    = (20,  25,  40)   # near-black     — body text (max readability)
C_MUTED   = (80,  95, 115)   # medium gray    — footer / sub-labels
C_KPI_VAL = (0,  120,  60)   # vivid green    — KPI values
C_KPI_BG  = (228, 241, 255)  # clear sky blue — KPI card background
C_RULE    = (150, 175, 210)  # medium blue    — divider lines


def _safe(text: str) -> str:
    text = str(text)
    for src, dst in [
        ('\u2014', '-'), ('\u2013', '-'), ('\u2018', "'"), ('\u2019', "'"),
        ('\u201c', '"'), ('\u201d', '"'), ('\u2022', '*'), ('\u25c8', '*'),
        ('\u203a', '>'), ('\u00d7', 'x'), ('\u2265', '>='), ('\u2264', '<='),
        ('\u00a0', ' '), ('\u2026', '...'),
    ]:
        text = text.replace(src, dst)
    return text.encode('latin-1', 'replace').decode('latin-1')


def _chart_to_png(fig_dict: dict) -> str | None:
    """Render a Plotly figure dict to a temporary PNG with a light theme for print."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        fig = go.Figure(fig_dict)
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='#f8fafc',
            font=dict(color='#1e293b', size=11),
            title_font=dict(color='#1e293b'),
            xaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#cbd5e1', color='#475569'),
            yaxis=dict(gridcolor='#e2e8f0', zerolinecolor='#cbd5e1', color='#475569'),
            legend=dict(bgcolor='white', bordercolor='#e2e8f0', font=dict(color='#475569')),
            margin=dict(t=40, b=30, l=50, r=20),
        )
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        pio.write_image(fig, tmp.name, format='png', width=700, height=280, scale=1.5)
        return tmp.name
    except Exception:
        return None


class AnalystReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(*C_ACCENT)
        self.cell(0, 8, 'ANALYST.AI - BUSINESS INTELLIGENCE REPORT', align='C', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(*C_RULE)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-13)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(*C_MUTED)
        self.cell(0, 8, f'Page {self.page_no()} | Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} | analyst.ai', align='C')

    def section_title(self, title: str):
        self.set_x(self.l_margin)
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(*C_ACCENT)
        self.ln(4)
        self.cell(0, 7, _safe(title.upper()), new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(*C_RULE)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def kpi_row(self, items: list[tuple[str, str]]):
        """Render KPI cards — never split value/label across a page break."""
        if not items:
            return
        if self.get_y() + 22 > self.page_break_trigger:
            self.add_page()
        n = min(len(items), 4)
        col_w = 188 // n
        start_x = self.l_margin
        y = self.get_y()
        for i, (label, value) in enumerate(items[:n]):
            x = start_x + i * col_w
            self.set_fill_color(*C_KPI_BG)
            self.set_draw_color(*C_RULE)
            self.rect(x, y, col_w - 2, 18, 'FD')
            self.set_font('Helvetica', 'B', 12)
            self.set_text_color(*C_KPI_VAL)
            self.set_xy(x + 2, y + 2)
            self.cell(col_w - 4, 7, _safe(str(value)[:18]), align='C')
            self.set_font('Helvetica', '', 6)
            self.set_text_color(*C_MUTED)
            self.set_xy(x + 2, y + 10)
            self.cell(col_w - 4, 6, _safe(str(label).upper()[:28]), align='C')
        self.set_xy(self.l_margin, y + 22)

    def body_text(self, text: str):
        if not text:
            return
        self.set_x(self.l_margin)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(*C_BODY)
        self.multi_cell(0, 5, _safe(text))
        self.ln(2)

    def insight_bullet(self, text: str):
        self.set_x(self.l_margin)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(*C_BODY)
        self.multi_cell(0, 5, _safe('  > ' + text))

    def metric_line(self, label: str, value: str):
        self.set_x(self.l_margin)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(*C_MUTED)
        self.cell(95, 5, _safe(str(label))[:45])
        self.set_text_color(*C_ACCENT)
        self.cell(0, 5, _safe(str(value))[:45], new_x='LMARGIN', new_y='NEXT')

    def embed_chart(self, img_path: str, caption: str = ''):
        """Embed a chart PNG. Starts a new page if insufficient space."""
        # Compute rendered height: image is 900×360px → aspect ~2.5
        # At 170mm wide → height ≈ 68mm. Add 6mm padding = 74mm needed.
        needed = 74
        if self.get_y() + needed > self.page_break_trigger:
            self.add_page()
        self.ln(3)
        try:
            self.image(img_path, x=self.l_margin + 9, w=170)
        except Exception:
            return
        if caption:
            self.set_x(self.l_margin)
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(*C_MUTED)
            self.cell(0, 4, _safe(caption), align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)


def _safe_scalar(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return None
    if isinstance(v, float):
        if v != v:  # NaN check
            return None
        return f'{v:,.2f}'
    if isinstance(v, int):
        return f'{v:,}'
    s = str(v).strip()
    return s if 1 <= len(s) < 50 else None


def generate_report(
    executive_summary: str,
    quality_report: dict,
    sections: dict,
    selected_analyses: list[str],
) -> str:
    pdf = AnalystReport()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(10, 12, 10)
    pdf.add_page()

    # ── Title block ───────────────────────────────────────────────────────────
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(*C_TITLE)
    pdf.cell(0, 12, 'ANALYST.AI', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 5, 'Automated Business Intelligence Report', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 5, f'Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}', align='C', new_x='LMARGIN', new_y='NEXT')
    sections_label = ', '.join(list(sections.keys())[:6])
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, _safe(f'Sections: {sections_label}'), align='C')
    pdf.ln(6)

    # ── Data Quality ──────────────────────────────────────────────────────────
    pdf.section_title('Data Quality')
    pdf.kpi_row([
        ('Total Records', f'{quality_report.get("total_rows", 0):,}'),
        ('Columns',       str(quality_report.get('total_cols', 0))),
        ('Quality Score', f'{quality_report.get("quality_score", 0)}/100'),
        ('Missing Data',  f'{quality_report.get("overall_null_pct", 0)}%'),
    ])

    # ── Executive Summary ─────────────────────────────────────────────────────
    pdf.section_title('Executive Summary')
    pdf.body_text(executive_summary)

    # ── Pre-render all charts in parallel ─────────────────────────────────────
    # Collect all (section_name, chart_key, fig_dict) tuples — max 2 per section
    render_tasks: list[tuple[str, str, dict]] = []
    for section_name, section in sections.items():
        if not section:
            continue
        charts = section.get('charts') or {}
        for chart_key, fig_dict in list(charts.items())[:2]:  # max 2 charts per section in PDF
            if fig_dict and isinstance(fig_dict, dict):
                render_tasks.append((section_name, chart_key, fig_dict))

    rendered: dict[tuple[str, str], str] = {}  # (section_name, chart_key) -> tmp_path
    if render_tasks:
        with ThreadPoolExecutor(max_workers=4) as pool:
            future_map = {
                pool.submit(_chart_to_png, fig_dict): (section_name, chart_key)
                for section_name, chart_key, fig_dict in render_tasks
            }
            for future in as_completed(future_map):
                key = future_map[future]
                try:
                    path = future.result()
                    if path:
                        rendered[key] = path
                except Exception:
                    pass

    # ── Dynamic sections ──────────────────────────────────────────────────────
    tmp_images: list[str] = list(rendered.values())

    for section_name, section in sections.items():
        if not section:
            continue

        pdf.section_title(section_name)
        metrics = section.get('metrics') or {}

        # AI findings
        ai_insights = metrics.get('ai_insights') or []
        if ai_insights:
            pdf.set_x(pdf.l_margin)
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_text_color(*C_ACCENT)
            pdf.cell(0, 5, 'AI FINDINGS', new_x='LMARGIN', new_y='NEXT')
            pdf.ln(1)
            for insight in ai_insights:
                pdf.insight_bullet(str(insight))
            pdf.ln(3)

        # Commentary
        commentary = section.get('commentary', '')
        if commentary:
            pdf.body_text(commentary)

        # Key scalar metrics as KPI cards (max 4)
        scalar_metrics = [
            (k.replace('_', ' ').title(), _safe_scalar(v))
            for k, v in metrics.items()
            if k not in ('ai_insights', 'numeric_cols', 'categorical_cols',
                         'outliers', 'skewness', 'top_correlations')
            and _safe_scalar(v) is not None
        ]
        if scalar_metrics:
            pdf.kpi_row([(lbl, val) for lbl, val in scalar_metrics[:4]])
            for lbl, val in scalar_metrics[4:10]:
                pdf.metric_line(lbl, val)

        # Charts — embed pre-rendered PNGs (max 2 per section)
        for chart_key in list((section.get('charts') or {}).keys())[:2]:
            img_path = rendered.get((section_name, chart_key))
            if img_path:
                caption = chart_key.replace('_', ' ').title()
                pdf.embed_chart(img_path, caption)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_x(pdf.l_margin)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(*C_MUTED)
    pdf.multi_cell(0, 4, 'This report was generated automatically by Analyst.ai. All insights should be reviewed by a qualified analyst before implementation.')

    out = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    out_name = out.name
    out.close()
    with open(out_name, 'wb') as f:
        f.write(pdf.output())

    # Clean up temporary chart images
    for p in tmp_images:
        try:
            os.unlink(p)
        except Exception:
            pass

    return out_name
