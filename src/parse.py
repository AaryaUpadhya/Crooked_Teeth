import json
import glob
import re

rows = []

for f in sorted(glob.glob("curves/degraded_seqs_rep0*.json")):
    with open(f) as file:
        data = json.load(file)

    rate = float(re.search(r"rate([\d.]+)\.json", f).group(1))

    rows.append((
        rate,
        data.get("mean_itl_ms"),
        data.get("p99_itl_ms"),
        data.get("mean_ttft_ms"),
        data.get("p99_ttft_ms"),
    ))

for r in rows:
    print("rate","mean_itl_ms","p99_itl_ms","mean_tft_ms","p99_ttft_ms")
    print(r)