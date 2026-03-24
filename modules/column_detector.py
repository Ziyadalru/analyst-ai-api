import json
import os
import pandas as pd
import numpy as np
from groq import Groq

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
_FAST = "llama-3.1-8b-instant"

# ── Column role definitions ───────────────────────────────────────────────────
ROLE_KEYWORDS = {
    'date':           ['date', 'order date', 'created', 'placed', 'time', 'ordered'],
    'ship_date':      ['ship date', 'shipping date', 'shipped', 'delivery date', 'delivered'],
    'revenue':        ['sales', 'revenue', 'amount', 'total', 'price', 'value', 'income'],
    'profit':         ['profit', 'net income', 'earnings', 'gain', 'margin amount'],
    'profit_ratio':   ['profit ratio', 'margin ratio', 'profit margin', 'net margin'],
    'cost':           ['cost', 'cogs', 'expense', 'spend', 'expenditure'],
    'quantity':       ['quantity', 'qty', 'units', 'volume', 'count', 'ordered quantity', 'wanted'],
    'discount':       ['discount', 'reduction', 'promo', 'rebate'],
    'category':       ['category', 'department', 'segment', 'type', 'class', 'group', 'division'],
    'sub_category':   ['sub category', 'subcategory', 'sub-category', 'product type'],
    'product':        ['product', 'item', 'sku', 'name', 'description', 'product name', 'part', 'board'],
    'supplier':       ['supplier', 'vendor', 'manufacturer', 'partner', 'source'],
    'customer':       ['customer', 'client', 'buyer', 'account', 'consumer'],
    'customer_id':    ['customer id', 'client id', 'account id', 'buyer id'],
    'region':         ['region', 'country', 'market', 'territory', 'zone', 'area'],
    'city':           ['city', 'town', 'municipality'],
    'state':          ['state', 'province', 'county'],
    'status':         ['status', 'delivery status', 'order status', 'fulfillment'],
    'risk':           ['risk', 'late', 'delay', 'flag', 'late delivery risk'],
    'rating':         ['rating', 'score', 'review', 'feedback', 'stars'],
    'days_real':      ['days for shipping real', 'actual days', 'real days', 'actual shipping'],
    'days_scheduled': ['days for shipment scheduled', 'scheduled days', 'planned days'],
    'payment':        ['payment', 'payment type', 'payment method', 'transaction type'],
    'order_id':       ['order id', 'order number', 'transaction id', 'invoice'],
}

ANALYSIS_COLUMN_REQUIREMENTS = {
    'Executive Dashboard': ['date'],
    'Financial':           ['revenue'],
    'Supply Chain':        ['status', 'quantity'],
    'Customer':            ['customer', 'revenue'],
    'Risk':                ['status', 'quantity'],
    'Data':                [],
    'Demand & Forecast':   ['date', 'quantity'],
}

ALL_ROLES = list(ROLE_KEYWORDS.keys())


# ── Pattern-based detection (fast fallback) ───────────────────────────────────
def detect_columns(df: pd.DataFrame) -> dict:
    detected = {}
    cols_lower = {c: c.lower() for c in df.columns}
    for role, keywords in ROLE_KEYWORDS.items():
        best = None
        for col, col_l in cols_lower.items():
            for kw in keywords:
                if kw in col_l:
                    if role in ('date', 'ship_date'):
                        try:
                            pd.to_datetime(df[col].dropna().iloc[:50], errors='raise')
                            best = col; break
                        except:
                            continue
                    elif role in ('revenue', 'profit', 'profit_ratio', 'cost',
                                  'quantity', 'discount', 'rating', 'days_real', 'days_scheduled'):
                        if pd.api.types.is_numeric_dtype(df[col]):
                            best = col; break
                    else:
                        best = col; break
            if best:
                break
        detected[role] = best
    return detected


# ── AI-powered detection ──────────────────────────────────────────────────────
_ROLES_JSON = json.dumps({r: None for r in ALL_ROLES}, indent=2)

def _build_prompt(df: pd.DataFrame) -> str:
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    try:
        sample = df.head(3).to_string(index=False, max_cols=20)
        # Cap prompt size — large Excel files can have huge cell values
        sample = sample[:1500]
    except Exception:
        sample = str(df.columns.tolist()[:30])

    return f"""You are a data analyst. Map the columns of this dataset to standard metric roles.

Column names and data types:
{json.dumps(dtypes)}

Sample rows (first 5):
{sample}

Map each column to a role. Use the EXACT column name from the dataset, or null if no column fits.

Roles to fill:
- date: transaction/order date
- ship_date: shipping or delivery date
- revenue: sales amount, price, revenue, sales price, value
- profit: profit or net income
- profit_ratio: profit margin percentage
- cost: cost or expenses
- quantity: quantity ordered, produced, sold, wanted qty, loaded qty
- discount: discount amount or percentage
- category: product category, type, PCB type, department
- sub_category: sub-category or product sub-type
- product: product name, item, SKU, PCB reference, part number
- supplier: supplier or vendor name
- customer: customer name or ID
- customer_id: numeric/text customer identifier
- region: geographic region or country
- city: city
- state: state or province
- status: order or delivery status
- risk: risk flag or late delivery indicator
- rating: quality rating or score
- days_real: actual processing or shipping days
- days_scheduled: scheduled or planned days
- payment: payment method
- order_id: order or transaction reference number

Also return:
- data_type: one of sales|manufacturing|logistics|hr|inventory|finance|other
- description: one sentence describing what this dataset contains
- suggested_analyses: array of analysis category names that are genuinely applicable to this dataset. Use names like "Executive Dashboard", "Financial", "Supply Chain", "Customer", "Risk", "Data", "Demand & Forecast" as a guide, but you may suggest other relevant categories if the data warrants it (e.g. "HR & Workforce", "Inventory", "Manufacturing", "Marketing")
- notes: any important observation (e.g. "This is manufacturing data — scrap and remake columns exist")

Always include "Data" in suggested_analyses.

Return ONLY valid JSON, no markdown, no explanation:
{{
  "data_type": "...",
  "description": "...",
  "columns": {{
    "date": null,
    "ship_date": null,
    "revenue": null,
    "profit": null,
    "profit_ratio": null,
    "cost": null,
    "quantity": null,
    "discount": null,
    "category": null,
    "sub_category": null,
    "product": null,
    "supplier": null,
    "customer": null,
    "customer_id": null,
    "region": null,
    "city": null,
    "state": null,
    "status": null,
    "risk": null,
    "rating": null,
    "days_real": null,
    "days_scheduled": null,
    "payment": null,
    "order_id": null
  }},
  "suggested_analyses": ["Data"],
  "notes": "..."
}}"""


def ai_detect_columns(df: pd.DataFrame) -> dict | None:
    """Call Groq to intelligently map columns and understand the dataset."""
    try:
        res = _groq.chat.completions.create(
            model=_FAST,
            messages=[{"role": "user", "content": _build_prompt(df)}],
            temperature=0.0,
            max_tokens=700,
            timeout=10.0,
        )
        raw = res.choices[0].message.content.strip()
        start, end = raw.find('{'), raw.rfind('}')
        if start == -1 or end == -1:
            return None
        result = json.loads(raw[start:end + 1])

        # Validate — only keep column mappings where name actually exists in df
        df_cols = set(df.columns.tolist())
        valid_cols = {}
        for role, col_name in result.get('columns', {}).items():
            valid_cols[role] = col_name if (col_name and col_name in df_cols) else None

        # Keep any non-empty string suggestions, always include Data
        suggested = [
            s for s in result.get('suggested_analyses', ['Data'])
            if s and isinstance(s, str)
        ]
        if not suggested:
            suggested = ['Data']
        elif 'Data' not in suggested:
            suggested.append('Data')

        return {
            'columns':            valid_cols,
            'data_type':          result.get('data_type', 'other'),
            'description':        result.get('description', ''),
            'suggested_analyses': suggested,
            'notes':              result.get('notes', ''),
        }
    except Exception:
        return None


# ── Smart detection: AI first, pattern matching as fallback ───────────────────
def detect_columns_smart(df: pd.DataFrame) -> tuple[dict, dict]:
    """Returns (cols_dict, ai_meta_dict)."""
    pattern_cols = detect_columns(df)
    ai_result    = ai_detect_columns(df)

    if not ai_result:
        return pattern_cols, {
            'data_type':          'other',
            'description':        '',
            'suggested_analyses': suggest_analyses(df, pattern_cols),
            'notes':              '',
        }

    # Merge: AI takes priority, pattern matching fills remaining gaps
    merged = pattern_cols.copy()
    for role, col in ai_result['columns'].items():
        if col is not None:
            merged[role] = col

    meta = {
        'data_type':          ai_result['data_type'],
        'description':        ai_result['description'],
        'suggested_analyses': ai_result['suggested_analyses'],
        'notes':              ai_result['notes'],
    }
    return merged, meta


# ── Suggest analyses (pattern-based fallback) ─────────────────────────────────
def suggest_analyses(df: pd.DataFrame, detected: dict) -> list:
    suggestions = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for analysis, required in ANALYSIS_COLUMN_REQUIREMENTS.items():
        if analysis == 'Data':
            if len(numeric_cols) >= 2:
                suggestions.append(analysis)
            continue
        if all(detected.get(req) for req in required):
            suggestions.append(analysis)
    if len(numeric_cols) >= 2 and 'Data' not in suggestions:
        suggestions.append('Data')
    return suggestions


# ── Data quality ──────────────────────────────────────────────────────────────
def get_data_quality(df: pd.DataFrame) -> dict:
    total = len(df)
    report = {
        'total_rows':         total,
        'total_cols':         len(df.columns),
        'duplicate_rows':     int(df.duplicated().sum()),
        'duplicate_pct':      round(df.duplicated().sum() / total * 100, 1),
        'columns_with_nulls': int((df.isnull().sum() > 0).sum()),
        'overall_null_pct':   round(df.isnull().sum().sum() / (total * len(df.columns)) * 100, 1),
        'numeric_cols':       len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols':   len(df.select_dtypes(include=['object']).columns),
        'quality_score':      None,
    }
    score = 100
    score -= min(30, report['overall_null_pct'] * 3)
    score -= min(20, report['duplicate_pct'] * 2)
    report['quality_score'] = max(0, round(score))
    report['quality_label'] = (
        'Excellent' if report['quality_score'] >= 85
        else 'Good'      if report['quality_score'] >= 70
        else 'Fair'      if report['quality_score'] >= 50
        else 'Poor'
    )
    return report
