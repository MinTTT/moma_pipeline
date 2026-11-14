import json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

NEW = Path(r'G:\python_code\Moma_pipeline\scripts_dev\_sandbox\cp_pub_new')
OLD = Path(r'G:\python_code\Moma_pipeline\scripts_dev\_sandbox\cp_pub_old')
SRC = Path(r'G:\python_code\Moma_pipeline\scripts_dev\_sandbox\SeqA_run\exported_data')

errors = []

# 1) file sets identical
fn_new = sorted(p.name for p in NEW.iterdir() if not p.name.startswith('.'))
fn_old = sorted(p.name for p in OLD.iterdir() if not p.name.startswith('.'))
assert fn_new == fn_old, (fn_new, fn_old)
print('1) file sets identical:', fn_new)

# 2) NEW vs OLD recalculated CSV: must be equal
a = pd.read_csv(next(NEW.glob('*_cells_df.csv')))
b = pd.read_csv(next(OLD.glob('*_cells_df.csv')))
assert list(a.columns) == list(b.columns)
for col in a.columns:
    if a[col].dtype.kind in 'fiu' and b[col].dtype.kind in 'fiu':
        if not np.allclose(a[col].astype(float), b[col].astype(float), rtol=1e-9, atol=1e-9, equal_nan=True):
            errors.append(f'CSV column {col!r}: new != old')
    elif not a[col].equals(b[col]):
        errors.append(f'CSV column {col!r}: new != old (non-numeric)')
if not errors: print(f'2) recalculated CSVs identical ({len(a)} rows x {len(a.columns)} cols): OK')

# 3) both differ from pipeline CSV in the same expected columns (recompute delta)
src_csv = pd.read_csv(next(SRC.glob('*_cells_df.csv')))
key = ['channels', 'time_point', 'cell_index']
diff_cols = {}
for label, pub in (('new', a), ('old', b)):
    s = src_csv.sort_values(key).reset_index(drop=True)
    p = pub.sort_values(key).reset_index(drop=True)
    cols = []
    for col in s.columns:
        if col not in p.columns: continue
        if s[col].dtype.kind in 'fiu' and p[col].dtype.kind in 'fiu':
            if not np.allclose(s[col].astype(float), p[col].astype(float), rtol=1e-9, atol=1e-9, equal_nan=True):
                cols.append(col)
        elif not s[col].equals(p[col]):
            cols.append(col)
    diff_cols[label] = sorted(cols)
print('3) recompute-vs-pipeline differing columns new:', diff_cols['new'])
if diff_cols['new'] != diff_cols['old']:
    errors.append(f'recompute delta differs between implementations: {diff_cols}')
else:
    print('   identical delta in old and new (expected recompute refresh of', len(diff_cols['new']), 'columns)')

# 4) published pkls: cell records equal (parse + compare normalized records)
def records(pth):
    d = pickle.load(open(pth, 'rb'))
    s = d['cells_sideChannel_image_dict']['fov_cells_pars_dict']
    out = {}
    for ch in s:
        for t in s[ch]:
            for rec in s[ch][t]:
                out[(ch, t, rec['cell_index'])] = json.dumps(rec, sort_keys=True, default=str)
    return out
rn, ro = records(next(NEW.glob('*_exported_data.pkl'))), records(next(OLD.glob('*_exported_data.pkl')))
if rn != ro:
    bad = [k for k in rn if rn[k] != ro.get(k)]
    errors.append(f'pkl cell records differ at {len(bad)} cells, first {bad[:3]}')
else:
    print(f'4) published pkl cell records identical ({len(rn)} records): OK')

# 5) summary JSONs identical after scrubbing run-specific + provenance fields
def scrub(d):
    if isinstance(d, dict):
        return {k: scrub(v) for k, v in d.items() if k not in
                ('source_path', 'source_sha256', 'output_dir', 'timestamp', 'backup_dir',
                 'transaction', 'argv', 'command', 'python', 'provenance', 'output_sha256')}
    if isinstance(d, list): return [scrub(x) for x in d]
    return d
sn = json.loads(next(NEW.glob('*_cell_parameters_summary.json')).read_text())
so = json.loads(next(OLD.glob('*_cell_parameters_summary.json')).read_text())
if scrub(sn) != scrub(so):
    errors.append('summary JSON differs after scrub')
    for k in sorted(set(sn) | set(so)):
        if scrub(sn.get(k)) != scrub(so.get(k)): print('   differs:', k)
else:
    print('5) summary JSONs identical after scrubbing run-specific/provenance fields: OK')

print()
if errors:
    print('PUBLISH PARITY FAILED:'); [print(' -', e) for e in errors]; sys.exit(1)
print('PUBLISH PARITY OK (new == old)')
