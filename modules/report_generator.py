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
    """Render a Plotly figure dict to PNG using matplotlib (no browser needed)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        traces = fig_dict.get('data', [])
        layout = fig_dict.get('layout', {})
        title = layout.get('title', '')
        if isinstance(title, dict):
            title = title.get('text', '')

        fig, ax = plt.subplots(figsize=(9, 3.5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8fafc')
        ax.grid(True, color='#e2e8f0', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        colors = ['#f59e0b', '#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#06b6d4']
        plotted = False

        for i, trace in enumerate(traces[:6]):
            c = colors[i % len(colors)]
            t = trace.get('type', 'bar')
            x = trace.get('x') or trace.get('labels') or []
            y = trace.get('y') or trace.get('values') or []
            z = trace.get('z')
            name = trace.get('name', '')

            try:
                # Histogram — only has x, compute bins with numpy
                if t == 'histogram':
                    if not x:
                        continue
                    ax.hist([float(v) for v in x if v is not None], bins=30, color=c, alpha=0.8, label=name)
                    plotted = True
                    continue

                # Heatmap — uses z matrix
                if t == 'heatmap':
                    if z is None:
                        continue
                    z_arr = np.array(z, dtype=float)
                    im = ax.imshow(z_arr, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
                    x_labels = trace.get('x') or []
                    y_labels = trace.get('y') or []
                    if x_labels:
                        ax.set_xticks(range(len(x_labels)))
                        ax.set_xticklabels([str(v)[:10] for v in x_labels], rotation=45, fontsize=6)
                    if y_labels:
                        ax.set_yticks(range(len(y_labels)))
                        ax.set_yticklabels([str(v)[:10] for v in y_labels], fontsize=6)
                    plt.colorbar(im, ax=ax, fraction=0.03)
                    plotted = True
                    continue

                # Filter None values from y (NaN → null → None after numpy_safe round-trip)
                y_clean = [float(v) for v in y if v is not None]
                if not y_clean:
                    continue

                x_clean = [v for v in x if v is not None]
                x_pos = list(range(len(y_clean))) if not x_clean else None

                if t in ('bar', 'waterfall', 'funnel'):
                    x_labels = [str(v)[:15] for v in x_clean] if x_clean else x_pos
                    ax.bar(x_labels, y_clean, color=c, alpha=0.85, label=name)
                elif t in ('scatter', 'line'):
                    mode = trace.get('mode', 'lines+markers' if t == 'line' else 'markers')
                    # Keep x numeric when possible (scatter requires numeric x)
                    try:
                        xs = [float(v) for v in x_clean] if x_clean else x_pos
                    except (TypeError, ValueError):
                        xs = [str(v) for v in x_clean] if x_clean else x_pos
                    if 'lines' in mode:
                        ax.plot(xs, y_clean, color=c, linewidth=2, label=name)
                    if 'markers' in mode:
                        ax.scatter(xs, y_clean, color=c, s=30, label=name)
                    if 'lines' not in mode and 'markers' not in mode:
                        ax.plot(xs, y_clean, color=c, linewidth=2, label=name)
                elif t == 'pie':
                    lbls = [str(v)[:12] for v in x_clean] if x_clean else [str(j) for j in range(len(y_clean))]
                    ax.pie(y_clean, labels=lbls, colors=colors[:len(y_clean)], autopct='%1.0f%%')
                elif t == 'box':
                    ax.boxplot(y_clean, patch_artist=True, boxprops=dict(facecolor=c, alpha=0.7))
                else:
                    x_labels = [str(v)[:15] for v in x_clean] if x_clean else x_pos
                    ax.bar(x_labels, y_clean, color=c, alpha=0.85, label=name)
                plotted = True
            except Exception:
                continue

        if not plotted:
            plt.close(fig)
            return None

        if title:
            ax.set_title(str(title)[:60], fontsize=10, color='#1e293b', pad=8)

        xaxis = layout.get('xaxis', {})
        yaxis = layout.get('yaxis', {})
        if isinstance(xaxis, dict) and xaxis.get('title'):
            xt = xaxis['title']
            ax.set_xlabel(xt.get('text', xt) if isinstance(xt, dict) else str(xt), fontsize=8)
        if isinstance(yaxis, dict) and yaxis.get('title'):
            yt = yaxis['title']
            ax.set_ylabel(yt.get('text', yt) if isinstance(yt, dict) else str(yt), fontsize=8)

        ax.tick_params(axis='x', labelsize=7, rotation=20)
        ax.tick_params(axis='y', labelsize=7)

        handles, labels = ax.get_legend_handles_labels()
        if handles and len(handles) > 1:
            ax.legend(fontsize=7, framealpha=0.7)

        plt.tight_layout(pad=1.0)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.close()
        fig.savefig(tmp.name, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
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
