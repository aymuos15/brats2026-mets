import json, os
os.chdir(os.path.expanduser("~/brats2025/localeval"))
alld = {}
for f in ["decode_scores.json", "decode_scores2.json", "blend_scores.json"]:
    try:
        alld.update(json.load(open(f)))
    except Exception:
        pass


def mean(v):
    return sum(v) / len(v)


rows = []
for k, v in alld.items():
    dsc = mean(v["dsc"]) if "dsc" in v else None
    nsd = mean(v["nsd"]) if "nsd" in v else None
    f1 = mean(v["f1"]) if "f1" in v else None
    rows.append((k, dsc, nsd, f1))

rows.sort(key=lambda r: (-(r[3] if r[3] is not None else -1),
                         -(r[1] if r[1] is not None else -1)))


def fmt(x):
    return ("%.4f" % x) if x is not None else "   -   "


print("%-34s %8s %8s %8s" % ("combo", "dscMean", "nsdMean", "f1Mean"))
for k, d, n, f in rows:
    star = "  <== 3way" if k.startswith("WIN 3way") or "3way-v1" in k else ""
    print("%-34s %8s %8s %8s%s" % (k, fmt(d), fmt(n), fmt(f), star))
