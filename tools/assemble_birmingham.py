#!/usr/bin/env python3
"""Regenerate data/birmingham_42_assembled.csv at the current engine (f=56).

Columns: date, panel (fixed observed, preserved from the old CSV),
engine_clear (khayt from the clear-sky run), engine_veiled (khayt from the
measured-veil run, --radiance 162.55 --led-fraction 0.15).

Clear run lives under birmingham_{date} or birmingham_rest_{date}.
Veiled run lives under birmingham_veil_full_{date}_... or
birmingham_veil_rest_{date}_... (radiance 162.55 suffix).
"""
import re, pathlib, csv, sys

RUNS = pathlib.Path.home() / "twilight/validation/criterion_runs"
OLD = pathlib.Path.home() / "twilight-papers/data/birmingham_42_assembled.csv"
VEIL_SUFFIX = "skyglow_radiance_162.55_led-fraction_0.15"

def khayt(path):
    if not path.exists():
        return None
    m = re.search(r"khayt al-abyad\).*?depression ([\d.]+)", path.read_text())
    return float(m.group(1)) if m else None

def clear_run(date):
    for stem in (f"birmingham_{date}.txt", f"birmingham_rest_{date}.txt"):
        p = RUNS / stem
        if p.exists():
            return p
    return None

def veil_run(date):
    for stem in (f"birmingham_veil_full_{date}_{VEIL_SUFFIX}.txt",
                 f"birmingham_veil_rest_{date}_{VEIL_SUFFIX}.txt"):
        p = RUNS / stem
        if p.exists():
            return p
    return None

rows = list(csv.DictReader(open(OLD)))
out = []
miss_c = miss_v = 0
for r in rows:
    d = r["date"]
    c = khayt(clear_run(d)) if clear_run(d) else None
    v = khayt(veil_run(d)) if veil_run(d) else None
    if c is None: miss_c += 1
    if v is None: miss_v += 1
    out.append((d, r["panel"], c, v))

write = "--write" in sys.argv
print(f"dates: {len(out)}  missing clear: {miss_c}  missing veiled: {miss_v}")
for d, p, c, v in out:
    print(f"  {d}  panel={p:>5}  clear={c if c is not None else '--':>6}  veil={v if v is not None else '--':>6}")
def stats():
    import statistics, math
    # winter veil dates (birmingham_veil_full set)
    WINTER = {"2015-01-11","2015-01-19","2015-01-24","2015-02-06","2015-02-18",
              "2015-02-22","2015-02-23","2015-02-24","2015-02-27","2015-11-13",
              "2015-11-28","2015-12-10","2015-12-19","2015-12-25"}
    SUMMER = {"2015-06-07","2015-06-22","2015-06-30","2015-07-06"}
    cr = [(c - float(p)) for d, p, c, v in out if c is not None]        # clear residuals
    eng = [c for d, p, c, v in out if c is not None]
    pan = [float(p) for d, p, c, v in out if c is not None]
    print("\n=== f=56 Birmingham stats ===")
    print(f"clear residual mean = {statistics.mean(cr):+.2f}, RMS = {math.sqrt(sum(x*x for x in cr)/len(cr)):.2f}, n={len(cr)}")
    if len(eng) > 1:
        mx, my = statistics.mean(eng), statistics.mean(pan)
        num = sum((a-mx)*(b-my) for a,b in zip(eng,pan))
        den = math.sqrt(sum((a-mx)**2 for a in eng)*sum((b-my)**2 for b in pan))
        print(f"Pearson r(engine_clear, panel) = {num/den:.2f}" if den else "r=NA")
    # veil residual + bracket over winter dates
    vr, brack, nwin = [], 0, 0
    for d, p, c, v in out:
        if d in WINTER and v is not None:
            vr.append(v - float(p)); nwin += 1
            if c is not None and min(c, v) <= float(p) <= max(c, v): brack += 1
    if vr:
        print(f"winter veil residual mean = {statistics.mean(vr):+.2f}, panel bracketed on {brack} of {nwin} winter dates")
    print("summer clear values: " + ", ".join(f"{d[5:]}={c}" for d,p,c,v in out if d in SUMMER and c is not None))

if write and miss_c == 0 and miss_v == 0:
    with open(OLD, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "panel", "engine_clear", "engine_veiled"])
        for d, p, c, v in out:
            w.writerow([d, p, f"{c:.2f}", f"{v:.2f}"])
    print(f"WROTE {OLD}")
    stats()
elif write:
    print(f"NOT WRITTEN: {miss_c} clear + {miss_v} veiled runs still missing")
elif miss_c == 0:
    stats()  # clear-sky stats available even without veil
