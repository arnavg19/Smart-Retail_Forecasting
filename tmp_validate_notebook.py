import json
from pathlib import Path
import pandas as pd

root = Path(r'd:\Smart-Rertail_Forcasting')
feature = root / 'data' / 'processed' / 'daily_demand_features.csv'
feature.parent.mkdir(parents=True, exist_ok=True)

dates = pd.date_range('2024-01-01', periods=30, freq='D')
rows = []
for item in ['A', 'B', 'C']:
    for d in dates:
        q = 1 if d.day % 5 == 0 else 0
        rows.append({
            'item_id': item,
            'invoice_date': d.strftime('%Y-%m-%d'),
            'quantity': q,
            'unit_price': 10 + (d.day % 7),
            'lag_1_days': 0,
            'lag_7_days': 0,
            'lag_14_days': 0,
            'lag_30_days': 0,
            'rolling_7d_mean_qty': 0,
            'rolling_30d_mean_qty': 0,
            'rolling_7d_std_qty': 0,
            'rolling_30d_std_qty': 0,
            'rolling_7d_min_qty': 0,
            'rolling_30d_min_qty': 0,
            'rolling_7d_max_qty': 0,
            'rolling_30d_max_qty': 0,
            'rolling_30d_avg_unit_price': 10,
            'relative_price_ratio': 1.0,
        })

pd.DataFrame(rows).to_csv(feature, index=False)
nb = json.loads((root / 'notebooks' / '02_eda_analysis.ipynb').read_text(encoding='utf-8'))
ns = {'__name__': '__main__'}
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        code = ''.join(cell.get('source', []))
        exec(code, ns, ns)
print('NOTEBOOK_EXEC_OK', 'rows=', len(rows), 'feature_exists=', feature.exists())
