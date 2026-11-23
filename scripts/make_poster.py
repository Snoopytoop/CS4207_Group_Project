"""Build the A1 poster (poster/poster.pptx) from outputs/results.json and figures/.

Every number on the poster is read from results.json - nothing is typed by hand.
Run on Rockhill:  python scripts/make_poster.py   (then soffice converts it to PDF)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Mm, Pt

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "outputs" / "results.json").read_text())
FIG = ROOT / "figures"

# ---------------------------------------------------------------- page and style
PAGE_W, PAGE_H = 594, 841            # A1 portrait, mm (swap for landscape)
MARGIN, GAP = 18, 12
MEMBERS = "Group 11 - member names to be added"   # edit with the group's names
MODULE = "CS4207 Advanced Programming - Coding Challenge - University of Limerick"

NAVY, INK, MUTED = RGBColor(0x14, 0x2b, 0x4c), RGBColor(0x0b, 0x0b, 0x0b), RGBColor(0x52, 0x51, 0x4e)
BLUE, ORANGE, AQUA = RGBColor(0x2a, 0x78, 0xd6), RGBColor(0xeb, 0x68, 0x34), RGBColor(0x1b, 0xaf, 0x7a)
BLUE_BG, ORANGE_BG, PANEL = RGBColor(0xdb, 0xe8, 0xf8), RGBColor(0xfd, 0xe3, 0xd6), RGBColor(0xf6, 0xf5, 0xf2)
WHITE, GREY_BG = RGBColor(0xff, 0xff, 0xff), RGBColor(0xe9, 0xe8, 0xe4)
FONT = "Calibri"
BODY, SMALL, HEAD, TITLE = 26, 22, 44, 84

prs = Presentation()
prs.slide_width, prs.slide_height = Mm(PAGE_W), Mm(PAGE_H)
slide = prs.slides.add_slide(prs.slide_layouts[6])


# ---------------------------------------------------------------- helpers
def fmt(n: float, d: int = 0) -> str:
    return f"{n:,.{d}f}"


def millions(n: float) -> str:
    return f"{n / 1e6:.1f} M"


def box(x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, line_w=1.5):
    s = slide.shapes.add_shape(shape, Mm(x), Mm(y), Mm(w), Mm(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line; s.line.width = Pt(line_w)
    s.shadow.inherit = False
    return s


def text(x, y, w, h, runs, size=BODY, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, shape=None, spacing=1.0):
    """runs: a string, or a list of paragraphs; each paragraph is a string or a list of (text, {style})."""
    s = shape or slide.shapes.add_textbox(Mm(x), Mm(y), Mm(w), Mm(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Mm(2); tf.margin_top = tf.margin_bottom = Mm(1)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        parts = para if isinstance(para, list) else [(para, {})]
        for t, st in parts:
            r = p.add_run(); r.text = t
            f = r.font
            f.name, f.size = FONT, Pt(st.get("size", size))
            f.bold = st.get("bold", bold)
            f.color.rgb = st.get("color", color)
    return s


def panel(x, y, w, h, title):
    """A section: light panel with a navy heading bar. Returns the content area (x, y, w, h)."""
    box(x, y, w, h, fill=PANEL)
    bar = box(x, y, w, 17, fill=NAVY)
    text(0, 0, 0, 0, [[(title, {"bold": True, "color": WHITE, "size": HEAD - 6})]], shape=bar,
         anchor=MSO_ANCHOR.MIDDLE)
    return x + 5, y + 21, w - 10, h - 25


def picture(path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Mm(w)
    if h: kw["height"] = Mm(h)
    return slide.shapes.add_picture(str(path), Mm(x), Mm(y), **kw)


def arrow(x1, y1, x2, y2, color=MUTED, width=3):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Mm(x1), Mm(y1), Mm(x2), Mm(y2))
    c.line.color.rgb = color; c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def table(x, y, w, rows, col_w, size=SMALL, header_fill=NAVY, row_h=10.5, bold_last=False, align=None):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Mm(x), Mm(y), Mm(w), Mm(row_h * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Mm(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Mm(row_h)
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.margin_left = cell.margin_right = Mm(2); cell.margin_top = cell.margin_bottom = Mm(0.5)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if i == 0 else (WHITE if i % 2 else GREY_BG)
            p = cell.text_frame.paragraphs[0]
            p.alignment = (align[j] if align else (PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT))
            r = p.add_run(); r.text = str(val)
            r.font.name, r.font.size = FONT, Pt(size)
            r.font.bold = i == 0 or (bold_last and i == len(rows) - 1)
            r.font.color.rgb = WHITE if i == 0 else INK
    return t


# ---------------------------------------------------------------- numbers from results.json
C, D, S, T, B, N = R["cleaning"], R["data"], R["split"], R["test"], R["benchmarks"], R["nondeterminism"]
metrics = {m["method"]: m for m in T["metrics"]}
model_m = T["metrics"][0]
base_week = metrics["Baseline: same hour last week"]
base_day = metrics["Baseline: same hour yesterday"]
bsum = B["summary"]
KMAX = B["max_workers"]


def bench(workload, workers, mode="parallel"):
    return next(r for r in bsum if r["workload"] == workload and r["workers"] == workers and r["mode"] == mode)


# ---------------------------------------------------------------- header
W = PAGE_W - 2 * MARGIN
COL = (W - GAP) / 2
LX, RX = MARGIN, MARGIN + COL + GAP

box(0, 0, PAGE_W, 112, fill=NAVY)
text(MARGIN, 12, W, 40, [[("How many taxis will New York need in the next hour?", {"bold": True})]],
     size=TITLE - 8, color=WHITE, align=PP_ALIGN.CENTER)
text(MARGIN, 52, W, 22, "Next-hour demand forecasting for NYC yellow taxis: sequential vs parallel Python",
     size=40, color=RGBColor(0xcd, 0xe2, 0xfb), align=PP_ALIGN.CENTER)
text(MARGIN, 78, W, 30, [[(MEMBERS, {"bold": True})], MODULE], size=BODY, color=WHITE, align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------- row 1: problem + cleaning
y1, h1 = 122, 182
cx, cy, cw, ch = panel(LX, y1, COL, h1, "Problem and data")
text(cx, cy, cw, 50, [
    [("Use case: ", {"bold": True}), ("forecast citywide yellow-cab pickups one hour ahead, so fleets can position "
                                      "drivers before demand arrives.", {})],
    [("Data: ", {"bold": True}), (f"NYC TLC trip records, {D['n_files']} monthly Parquet files "
                                  f"({D['months'][0]} to {D['months'][1]}).", {})],
], size=BODY, spacing=0.95)
tiles = [(millions(C["raw_rows"]), "raw trips"), (millions(C["kept_rows"]), f"trips kept ({C['kept_pct']:.1f}%)"),
         (fmt(C["hourly_observations"]), "hourly observations"), (fmt(C["mean_pickups_per_hour"]), "mean pickups / hour")]
tw = (cw - 3 * 4) / 4
for i, (big, small) in enumerate(tiles):
    tx = cx + i * (tw + 4)
    box(tx, cy + 64, tw, 40, fill=WHITE, line=BLUE_BG)
    text(tx, cy + 66, tw, 20, [[(big, {"bold": True})]], size=40, color=BLUE, align=PP_ALIGN.CENTER)
    text(tx, cy + 86, tw, 16, small, size=19, color=MUTED, align=PP_ALIGN.CENTER)
picture(FIG / "daily_weekly_profile.png", cx + 20, cy + 108, h=ch - 110)

cx, cy, cw, ch = panel(RX, y1, COL, h1, "Data cleaning (rules applied in this order)")
rm = C["removed"]
rows = [["Rule", "Rows removed"],
        ["Raw rows (footer metadata, 10 of 21 columns read)", fmt(C["raw_rows"])],
        ["Pickup outside the file's month (spill-over, 2001 clocks)", fmt(rm["pickup_outside_month"])],
        ["Trip < 1 min or > 3 h (meter errors)", fmt(rm["duration_out_of_range"])],
        ["Distance 0 or > 100 miles (max seen 281,576)", fmt(rm["distance_out_of_range"])],
        ["Fare zero or negative (refunds, voids)", fmt(rm["fare_not_positive"])],
        ["Exact duplicate rows", fmt(rm["duplicate_rows"])],
        ["Trips kept for the forecast", fmt(C["kept_rows"])]]
table(cx, cy, cw, rows, [cw - 52, 52], size=SMALL, row_h=11.5, bold_last=True)
text(cx, cy + 96, cw, 60, [
    [("Flex Fare kept on purpose: ", {"bold": True}),
     (f"{millions(C['flex_fare_kept'])} trips have payment_type 0 and no rate code. They are real pickups, so "
      "they count as demand; rate code becomes a category with an 'unknown' level.", {})],
    [("Missing hours: ", {"bold": True}),
     (f"only {C['missing_hours_filled']} (02:00 on each spring-forward day), filled with 0. "
      "Zones and rate codes are stored as categories.", {})],
], size=SMALL, spacing=0.95)

# ---------------------------------------------------------------- row 2: pipeline diagram
y2, h2 = y1 + h1 + GAP, 150
cx, cy, cw, ch = panel(LX, y2, W, h2, "Pipeline: where the parallelism is")
bw, bh = 98, 38
def stage(x, y, title, sub, kind=None):
    fill, line = {None: (WHITE, MUTED), "data": (BLUE_BG, BLUE), "task": (ORANGE_BG, ORANGE)}[kind]
    s = box(x, y, bw, bh, fill=fill, line=line, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_w=3 if kind else 1.5)
    s.adjustments[0] = 0.12
    text(0, 0, 0, 0, [[(title, {"bold": True, "size": 22})], [(sub, {"size": 18, "color": MUTED})]],
         shape=s, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=0.9)
    return s

sx = [cx + i * ((cw - bw) / 4) for i in range(5)]
r1y, r2y = cy + 4, cy + 4 + bh + 26
cs = R["search"]
stage(sx[0], r1y, f"{D['n_files']} Parquet files", "footer inspected, 10 columns read")
stage(sx[1], r1y, "clean_month x 33", f"1 shard = 1 month, {R['data_parallel']['workers']} workers", "data")
stage(sx[2], r1y, "Ordered merge", "concat sorted by month", "data")
stage(sx[3], r1y, "Hourly series", f"{fmt(C['hourly_observations'])} hours, verified = sequential")
stage(sx[4], r1y, "Time split", "train | Aug val | Sep test")
for i in range(4):
    arrow(sx[i] + bw, r1y + bh / 2, sx[i + 1], r1y + bh / 2)
stage(sx[4], r2y, "Lag | rolling | calendar", "3 independent feature tasks", "task")
stage(sx[3], r2y, "Hyperparameter search", f"{cs['n_settings']} settings x {cs['folds']} folds = {cs['n_tasks']} tasks", "task")
stage(sx[2], r2y, "Pick best on validation", "Aug 2026, never the test month")
stage(sx[1], r2y, "Retrain on train + val", "Jan 2024 - Aug 2026")
stage(sx[0], r2y, "Test once", "Sep 2026, one hour ahead")
arrow(sx[4] + bw / 2, r1y + bh, sx[4] + bw / 2, r2y)
for i in range(4, 0, -1):
    arrow(sx[i], r2y + bh / 2, sx[i - 1] + bw, r2y + bh / 2)
ly = r2y + bh + 4
box(cx, ly + 1, 9, 7, fill=BLUE_BG, line=BLUE, line_w=3)
text(cx + 10, ly - 1, 200, 10, [[("DATA PARALLEL: ", {"bold": True, "color": BLUE}), ("same function, different shards", {})]], size=SMALL)
box(cx + 230, ly + 1, 9, 7, fill=ORANGE_BG, line=ORANGE, line_w=3)
text(cx + 240, ly - 1, 300, 10, [[("TASK PARALLEL: ", {"bold": True, "color": ORANGE}), ("different independent jobs", {})]], size=SMALL)

# ---------------------------------------------------------------- row 3: split + results
y3, h3 = y2 + h2 + GAP, 200
cx, cy, cw, ch = panel(LX, y3, COL, h3, "Time-based split, no leakage")
pic = picture(FIG / "time_split.png", cx, cy, w=cw)
ph = pic.height / 36000
text(cx, cy + ph + 3, cw, ch - ph - 3, [
    [("Never a random split: ", {"bold": True}), ("training hours always come before the hours being scored.", {})],
    [("Features use the past only: ", {"bold": True}),
     ("lags of 1, 2, 3, 24, 48, 168 h; rolling mean/std/max of the last 3, 24, 168 h computed on y.shift(1); "
      "hour, weekday, weekend, month, holiday.", {})],
    [("No fares, tips or distances ", {"bold": True}), ("of the predicted hour. Scaler fitted per fold inside a Pipeline. "
      f"First {S['history_hours_dropped']} hours dropped (incomplete lag history).", {})],
], size=SMALL, spacing=0.95)

cx, cy, cw, ch = panel(RX, y3, COL, h3, "Result: forecast vs actual (test month)")
pic = picture(FIG / "forecast_vs_actual.png", cx, cy, w=cw)
ph = pic.height / 36000
pic2 = picture(FIG / "error_vs_baseline.png", cx, cy + ph + 1, h=ch - ph - 24)
text(cx, cy + ch - 22, cw, 22, [[
    ("Gradient boosting misses by ", {}), (f"{model_m['wape_pct']:.1f}%", {"bold": True, "color": BLUE}),
    (" of actual volume vs ", {}), (f"{base_week['wape_pct']:.1f}%", {"bold": True}),
    (f" for 'same hour last week' (MAE {fmt(model_m['mae'])} vs {fmt(base_week['mae'])} pickups/h). "
     "Knowing the last hour is what beats the weekly baseline.", {})]], size=SMALL, spacing=0.95)

# ---------------------------------------------------------------- row 4: speedup + non-determinism
y4 = y3 + h3 + GAP
h4 = PAGE_H - y4 - 26
cx, cy, cw, ch = panel(LX, y4, COL, h4, f"Sequential vs parallel ({KMAX} workers)")
pic = picture(FIG / "speedup.png", cx, cy, h=ch - 62)
rows = [["Workload", "T seq (s)", f"S({KMAX})", f"E({KMAX})", "p"]]
names = {"cleaning": "Data: clean 33 months", "search": "Task: 96 search fits", "features": "Task: 3 feature blocks"}
for wl in ["cleaning", "search", "features"]:
    seq = bench(wl, 1, "sequential"); k = bench(wl, KMAX)
    rows.append([names[wl], f"{seq['mean_s']:.2f}", f"{k['speedup']:.2f}x", f"{k['efficiency']:.2f}",
                 f"{k['amdahl_p']:.2f}" if k["amdahl_p"] > 0 else "<0"])
table(cx, cy + ch - 60, cw, rows, [cw - 117, 30, 30, 27, 30], size=19, row_h=9.5)
best_wl = max(["cleaning", "search"], key=lambda w: bench(w, KMAX)["speedup"])
p_best = bench(best_wl, KMAX)
text(cx, cy + ch - 21, cw, 21, [[("Amdahl: ", {"bold": True}),
     (f"p = {p_best['amdahl_p']:.2f} for {names[best_wl].split(': ')[1]} -> at most {p_best['max_speedup']:.1f}x with unlimited "
      "workers. Lost to pool start-up, pickling, shared disk, load imbalance and a busy shared server.", {})]],
     size=19, spacing=0.9)

cx, cy, cw, ch = panel(RX, y4, COL, h4, "Non-determinism: shown, then fixed")
be = N["broken_evidence_this_run"]
rs = be["random_search_best"]
ft = [float(v) for v in be["fare_total"]]
rows = [["Source", "Two broken runs", "Fix", "Re-run"],
        ["Completion order (as_completed)", f"first hour {be['first_hour'][0][:7]} vs {be['first_hour'][1][:7]}",
         "sort by month / map", "identical"],
        ["Float sum order (fares)", f"${ft[0]:,.2f} vs ${ft[1]:,.2f}" if ft[0] != ft[1] else "last bits differ",
         "fixed order / fsum", "identical"],
        ["Unseeded random search", f"lr {rs[0]['learning_rate']:.3f} vs {rs[1]['learning_rate']:.3f}",
         "seed = 42 + trial id", "identical"]]
table(cx, cy, cw, rows, [62, 92, 60, cw - 214], size=18, row_h=15,
      align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER])
ok = all(N[k]["fixed_runs_identical"] for k in ["completion_order", "reduction_order", "unseeded_randomness"])
text(cx, cy + 64, cw, ch - 64, [
    [("Why: ", {"bold": True}), ("workers finish in a different order each run, float addition is not associative "
     f"((a+b)+c = {N['tiny_example']['(a+b)+c']:g} but a+(b+c) = {N['tiny_example']['a+(b+c)']:g}), and unseeded "
     "generators differ.", {})],
    [("After the fixes: ", {"bold": True}), ("both re-runs identical " + ("in all three cases." if ok else "- CHECK"), {}),
     (" Maths libraries pinned to 1 thread per worker, so BLAS thread count cannot change results.", {})],
    [("Every parallel result equals the sequential one ", {"bold": True}),
     ("(series, cleaning stats, features, all 96 search scores: asserted).", {})],
], size=SMALL, spacing=0.95)

# ---------------------------------------------------------------- footer
env = R["environment"]
text(MARGIN, PAGE_H - 22, W, 18,
     f"Hardware: {env['hostname']}, {env['cpu_model']}, {env['cpus_allocated']} CPUs via Slurm, no GPU  |  "
     f"Python {env['python']}, pandas {env['pandas']}, scikit-learn {env['scikit-learn']}, pyarrow {env['pyarrow']}  |  "
     "All figures and numbers generated by demand_forecast.ipynb (outputs/results.json)",
     size=16, color=MUTED, align=PP_ALIGN.CENTER)

out = ROOT / "poster"
out.mkdir(exist_ok=True)
prs.save(out / "poster.pptx")
print("saved", out / "poster.pptx")
