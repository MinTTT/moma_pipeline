"""Old-vs-new pipeline parity check on identical fov_0 copies.

Compares: cells_df.csv (schema + values), exported_data.pkl (info_dict,
morphology provenance, masks, channel images, timing, backgrounds, cell
records), and channel_info.txt.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path(r"G:\python_code\Moma_pipeline\scripts_dev\_sandbox\SeqA_old\exported_data")
NEW = Path(r"G:\python_code\Moma_pipeline\scripts_dev\_sandbox\SeqA_run\exported_data")


def find_one(d: Path, suffix: str) -> Path:
    hits = sorted(d.glob("*" + suffix))
    assert len(hits) == 1, f"expected exactly one {suffix} in {d}, got {hits}"
    return hits[0]


def norm_cell_record(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, np.ndarray):
            out[k] = ("array", v.shape, v.dtype.str, float(np.nan_to_num(np.asarray(v, dtype=float))).__round__(9) if v.dtype.kind in "fiu" else hash(v.tobytes()))
        elif isinstance(v, (list, tuple)):
            out[k] = ("seq", type(v).__name__, [float(x) if isinstance(x, (int, float, np.floating)) else str(x) for x in v][:60])
        elif isinstance(v, dict):
            out[k] = ("dict", sorted(v.keys()))
        else:
            out[k] = v
    return out


errors = []

# ---------- 1) CSV parity ----------
csv_old = pd.read_csv(find_one(OLD, "_cells_df.csv"))
csv_new = pd.read_csv(find_one(NEW, "_cells_df.csv"))
print(f"CSV rows: old={len(csv_old)} new={len(csv_new)}")
print(f"CSV cols: old={len(csv_old.columns)} new={len(csv_new.columns)}")
if set(csv_old.columns) != set(csv_new.columns):
    errors.append(f"CSV column mismatch: only-old={set(csv_old.columns)-set(csv_new.columns)} only-new={set(csv_new.columns)-set(csv_old.columns)}")
key = ["channels", "time_point", "cell_index"]
a = csv_old.sort_values(key).reset_index(drop=True)
b = csv_new.sort_values(key).reset_index(drop=True)
if len(a) != len(b):
    errors.append(f"CSV row count mismatch {len(a)} vs {len(b)}")
else:
    for col in a.columns:
        if col not in b.columns:
            continue
        sa, sb = a[col], b[col]
        if sa.dtype.kind in "fiu" and sb.dtype.kind in "fiu":
            if not np.allclose(sa.astype(float), sb.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True):
                bad = np.where(~np.isclose(sa.astype(float), sb.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True))[0]
                errors.append(f"CSV column {col!r} differs at rows {bad[:5]} (n={len(bad)})")
        else:
            if not sa.equals(sb):
                errors.append(f"CSV column {col!r} differs (non-numeric)")

# ---------- 2) PKL parity ----------
p_old = pickle.load(open(find_one(OLD, "_exported_data.pkl"), "rb"))
p_new = pickle.load(open(find_one(NEW, "_exported_data.pkl"), "rb"))
io_, in_ = p_old["info_dict"], p_new["info_dict"]
for k in ("time_points", "opticalChannels", "other_channels",
          "cell_detection_opticalChannel_name", "channel_keys", "algorithm"):
    if io_.get(k) != in_.get(k):
        errors.append(f"info_dict[{k!r}] mismatch")
mo, mn = io_.get("morphology", {}), in_.get("morphology", {})
if mo != mn:
    print("morphology old:", {k: mo.get(k) for k in list(mo)[:8]})
    print("morphology new:", {k: mn.get(k) for k in list(mn)[:8]})
    errors.append("info_dict.morphology differs")

s_old = p_old["cells_sideChannel_image_dict"]
s_new = p_new["cells_sideChannel_image_dict"]
if set(s_old) != set(s_new):
    errors.append(f"sideChannel keys differ: {set(s_old)^set(s_new)}")

for section in ("fov_cells_mask_dict", "fov_channels_image_dict"):
    so, sn = s_old[section], s_new[section]
    if sorted(so) != sorted(sn):
        errors.append(f"{section}: channel keys differ")
        continue
    n_bad = 0
    for ch in so:
        if sorted(so[ch]) != sorted(sn[ch]):
            errors.append(f"{section}[{ch!r}]: time keys differ")
            continue
        for t in so[ch]:
            vo, vn = so[ch][t], sn[ch][t]
            if isinstance(vo, dict):
                for oc in vo:
                    if not np.array_equal(np.asarray(vo[oc]), np.asarray(vn[oc])):
                        n_bad += 1
            elif not np.array_equal(np.asarray(vo), np.asarray(vn)):
                n_bad += 1
    if n_bad:
        errors.append(f"{section}: {n_bad} arrays differ")
    else:
        print(f"{section}: all arrays identical")

if s_old["time_points_seconds"] != s_new["time_points_seconds"]:
    errors.append("time_points_seconds differ")
else:
    print("time_points_seconds: identical")
if s_old["images_background_fluorescence"] != s_new["images_background_fluorescence"]:
    errors.append("images_background_fluorescence differ")
else:
    print("images_background_fluorescence: identical")

# cell parameter records
po, pn = s_old["fov_cells_pars_dict"], s_new["fov_cells_pars_dict"]
if sorted(po) != sorted(pn):
    errors.append("fov_cells_pars_dict: channel keys differ")
else:
    n_rec = 0
    bad_rec = 0
    for ch in po:
        for t in po[ch]:
            ro, rn = po[ch][t], pn[ch][t]
            if len(ro) != len(rn):
                errors.append(f"cell count differs at {ch!r}/{t!r}: {len(ro)} vs {len(rn)}")
                continue
            for rec_o, rec_n in zip(ro, rn):
                n_rec += 1
                if norm_cell_record(rec_o) != norm_cell_record(rec_n):
                    bad_rec += 1
                    if bad_rec <= 3:
                        diffkeys = [k for k in set(rec_o) | set(rec_n)
                                    if norm_cell_record(rec_o).get(k) != norm_cell_record(rec_n).get(k)]
                        print(f"  record diff at {ch!r}/{t!r} cell {rec_o.get('cell_index')}: keys {diffkeys[:8]}")
    print(f"cell records compared: {n_rec}, differing: {bad_rec}")
    if bad_rec:
        errors.append(f"{bad_rec} cell records differ")

# ---------- 3) channel_info.txt ----------
tx_old = find_one(OLD, "channel_info.txt").read_text()
tx_new = find_one(NEW, "channel_info.txt").read_text()
if tx_old != tx_new:
    errors.append("channel_info.txt differs")
else:
    print("channel_info.txt: identical")

# ---------- verdict ----------
print()
if errors:
    print("PARITY FAILED:")
    for e in errors:
        print("  -", e)
    sys.exit(1)
print("PARITY OK - new pipeline output matches the old pipeline bit-for-bit")
