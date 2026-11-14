import json
from pathlib import Path

new = json.loads(Path(r'G:\python_code\Moma_pipeline\scripts_dev\_sandbox\cp_out\fov_0_cell_parameters_error.json').read_text())
old = json.loads(Path(r'G:\python_code\Moma_pipeline\scripts_dev\_sandbox\cp_out_old\fov_0_cell_parameters_error.json').read_text())

def scrub(d):
    d = dict(d)
    d.pop("source_path", None); d.pop("source_sha256", None)
    d.pop("output_dir", None)
    return d

a, b = scrub(old), scrub(new)
print("old error head:", str(old.get("error"))[:120])
print("new error head:", str(new.get("error"))[:120])
a.pop("error", None); b.pop("error", None)

if a == b:
    print("ERROR REPORTS IDENTICAL (modulo paths)")
else:
    ka, kb = set(a), set(b)
    print("key diff:", ka ^ kb)
    for k in sorted(ka & kb):
        if a[k] != b[k]:
            print(f"  differs: {k}")
            if k == "invalid_cells":
                print("   old n:", len(a[k]), "new n:", len(b[k]))
                for ra, rb in zip(a[k], b[k]):
                    if ra != rb:
                        diff = [x for x in set(ra) | set(rb) if ra.get(x) != rb.get(x)]
                        print("   first differing record keys:", diff)
                        break
