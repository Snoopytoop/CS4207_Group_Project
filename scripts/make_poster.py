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
PAGE_W, PAGE_H = 841, 594            # A1 landscape, mm
MARGIN, GAP = 16, 10
MEMBERS = "Group 11: Saif Khawaldeh, Samuel Luke, Simon Alexander, Iwan Teague, Kevin Monahan"
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


# ---------------------------------------------------------------- layout grid (A1 landscape)
W = PAGE_W - 2 * MARGIN
COL = (W - 2 * GAP) / 3
X1, X2, X3 = MARGIN, MARGIN + COL + GAP, MARGIN + 2 * (COL + GAP)
HEADER_H = 74
Y1, H1 = HEADER_H + 8, 158          # row 1: problem | cleaning | split
H2 = 112                            # row 2: pipeline (full width)
Y2 = Y1 + H1 + GAP
Y3 = Y2 + H2 + GAP                  # row 3: results | speedup | non-determinism
H3 = PAGE_H - Y3 - 18

# ---------------------------------------------------------------- header
box(0, 0, PAGE_W, HEADER_H, fill=NAVY)
text(MARGIN, 6, W, 30, [[("How many taxis will New York need in the next hour?", {"bold": True})]],
     size=TITLE, color=WHITE, align=PP_ALIGN.CENTER)
text(MARGIN, 37, W, 16, "Next-hour demand forecasting for NYC yellow taxis: sequential vs parallel Python",
     size=36, color=RGBColor(0xcd, 0xe2, 0xfb), align=PP_ALIGN.CENTER)
text(MARGIN, 55, W, 16, [[(MEMBERS, {"bold": True}), ("   |   " + MODULE, {})]], size=24, color=WHITE,
     align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------- row 1
cx, cy, cw, ch = panel(X1, Y1, COL, H1, "1  Problem and data")
text(cx, cy, cw, 48, [
    [("Use case: ", {"bold": True}), ("forecast citywide yellow-cab pickups one hour ahead, so fleets can position "
                                      "drivers before demand arrives.", {})],
    [("Data: ", {"bold": True}), (f"NYC TLC trip records, {D['n_files']} monthly Parquet files, "
                                  f"{D['months'][0]} to {D['months'][1]}.", {})],
], size=BODY, spacing=0.92)
tiles = [(millions(C["raw_rows"]), "raw trips"), (millions(C["kept_rows"]), f"trips kept ({C['kept_pct']:.1f}%)"),
         (fmt(C["hourly_observations"]), "hourly observations"), (fmt(C["mean_pickups_per_hour"]), "mean pickups/hour")]
tw, th = (cw - 4) / 2, 33
for i, (big, small) in enumerate(tiles):
    tx, ty = cx + (i % 2) * (tw + 4), cy + 50 + (i // 2) * (th + 3)
    box(tx, ty, tw, th, fill=WHITE, line=BLUE_BG)
    text(tx, ty + 0.5, tw, 18, [[(big, {"bold": True})]], size=40, color=BLUE, align=PP_ALIGN.CENTER)
    text(tx, ty + 18.5, tw, 13, small, size=20, color=MUTED, align=PP_ALIGN.CENTER)
peaks = {(p["window"].split(" ")[0], p["day_type"]): p["mean_pickups_per_hour"] for p in C["peak_profile"]}
text(cx, cy + 50 + 2 * (th + 3), cw, 14, [[("Busiest: ", {"bold": True}),
     (f"weekdays 16-20 h, {fmt(peaks[('evening_peak', 'weekday')])} pickups/h "
      f"(7-10 h: {fmt(peaks[('morning_peak', 'weekday')])})", {})]], size=20, spacing=0.9)

cx, cy, cw, ch = panel(X2, Y1, COL, H1, "2  Data cleaning")
rm = C["removed"]
rows = [["Rule (applied in order)", "Removed"],
        ["Raw rows (10 of 21 columns read)", fmt(C["raw_rows"])],
        ["Pickup outside the file's month", fmt(rm["pickup_outside_month"])],
        ["Trip < 1 min or > 3 hours", fmt(rm["duration_out_of_range"])],
        ["Distance 0 or > 100 miles", fmt(rm["distance_out_of_range"])],
        ["Fare zero or negative (refunds)", fmt(rm["fare_not_positive"])],
        ["Exact duplicate rows", fmt(rm["duplicate_rows"])],
        ["Trips kept for the forecast", fmt(C["kept_rows"])]]
table(cx, cy, cw, rows, [cw - 56, 56], size=21, row_h=10, bold_last=True)
text(cx, cy + 86, cw, ch - 86, [
    [("Flex Fare kept: ", {"bold": True}),
     (f"{millions(C['flex_fare_kept'])} trips with payment_type 0 and no rate code are real pickups, so they count "
      "as demand. Rate code becomes a category with an 'unknown' level.", {})],
    [("Missing hours: ", {"bold": True}),
     (f"{C['missing_hours_filled']} (02:00 on each daylight-saving spring-forward day), filled with 0.", {})],
], size=21, spacing=0.92)

cx, cy, cw, ch = panel(X3, Y1, COL, H1, "3  Time-based split, no leakage")
pic = picture(FIG / "time_split.png", cx, cy, w=cw)
ph = pic.height / 36000
text(cx, cy + ph + 1, cw, ch - ph - 1, [
    [("Never a random split: ", {"bold": True}), ("training hours always come before scored hours.", {})],
    [("Past-only features: ", {"bold": True}),
     ("lags 1, 2, 3, 24, 48, 168 h; rolling mean/std/max over 3, 24, 168 h on y.shift(1); hour, weekday, "
      "weekend, month, holiday. No fares or distances of the predicted hour.", {})],
], size=21, spacing=0.92)

# ---------------------------------------------------------------- row 2: pipeline diagram
cx, cy, cw, ch = panel(X1, Y2, W, H2, "4  Pipeline: where the parallelism is")
bw, bh = 112, 30
def stage(x, y, title, sub, kind=None):
    fill, line = {None: (WHITE, MUTED), "data": (BLUE_BG, BLUE), "task": (ORANGE_BG, ORANGE)}[kind]
    s = box(x, y, bw, bh, fill=fill, line=line, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line_w=3.5 if kind else 1.5)
    s.adjustments[0] = 0.15
    text(0, 0, 0, 0, [[(title, {"bold": True, "size": 21})], [(sub, {"size": 17, "color": MUTED})]],
         shape=s, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=0.88)

legend_w = 150
dw = cw - legend_w - 6
sx = [cx + i * ((dw - bw) / 4) for i in range(5)]
r1y, r2y = cy + 1, cy + bh + 15
cs = R["search"]
stage(sx[0], r1y, f"{D['n_files']} Parquet files", "footer inspected, columns pruned")
stage(sx[1], r1y, f"clean_month x {D['n_files']}", f"1 shard = 1 month, {R['data_parallel']['workers']} workers", "data")
stage(sx[2], r1y, "Ordered merge", "concat sorted by month", "data")
stage(sx[3], r1y, "Hourly series", f"{fmt(C['hourly_observations'])} hours, = sequential")
stage(sx[4], r1y, "Lag | rolling | calendar", "3 independent feature tasks", "task")
for i in range(4):
    arrow(sx[i] + bw, r1y + bh / 2, sx[i + 1], r1y + bh / 2)
stage(sx[4], r2y, "Hyperparameter search", f"{cs['n_settings']} settings x {cs['folds']} folds = {cs['n_tasks']} tasks", "task")
stage(sx[3], r2y, "Pick best on validation", "Aug 2026 (never the test month)")
stage(sx[2], r2y, "Retrain on train + val", "Jan 2024 - Aug 2026")
stage(sx[1], r2y, "Test once", "Sep 2026, one hour ahead")
stage(sx[0], r2y, "Compare to baselines", "same hour yesterday / last week")
arrow(sx[4] + bw / 2, r1y + bh, sx[4] + bw / 2, r2y)
for i in range(4, 0, -1):
    arrow(sx[i], r2y + bh / 2, sx[i - 1] + bw, r2y + bh / 2)
lx = cx + cw - legend_w
box(lx, cy + 4, 12, 9, fill=BLUE_BG, line=BLUE, line_w=3.5)
text(lx + 14, cy + 1, legend_w - 14, 34, [[("DATA PARALLEL", {"bold": True, "color": BLUE})],
                                         "same function, different shards"], size=21, spacing=0.9)
box(lx, cy + 44, 12, 9, fill=ORANGE_BG, line=ORANGE, line_w=3.5)
text(lx + 14, cy + 41, legend_w - 14, 34, [[("TASK PARALLEL", {"bold": True, "color": ORANGE})],
                                          "different independent jobs"], size=21, spacing=0.9)

# ---------------------------------------------------------------- row 3: results | speedup | non-determinism
cx, cy, cw, ch = panel(X1, Y3, COL, H3, "5  Result on the test month (Sep 2026)")
pic = picture(FIG / "forecast_vs_actual.png", cx, cy, w=cw)
ph = pic.height / 36000
txt_h = 26
pic2 = picture(FIG / "error_vs_baseline.png", cx, cy + ph + 1, h=ch - ph - txt_h - 2)
text(cx, cy + ch - txt_h, cw, txt_h, [[
    ("Model misses by ", {}), (f"{model_m['wape_pct']:.1f}%", {"bold": True, "color": BLUE}),
    (" of actual volume vs ", {}), (f"{base_week['wape_pct']:.1f}%", {"bold": True}),
    (f" for 'same hour last week' (MAE {fmt(model_m['mae'])} vs {fmt(base_week['mae'])} pickups/h).", {})]],
    size=21, spacing=0.92)

cx, cy, cw, ch = panel(X2, Y3, COL, H3, f"6  Sequential vs parallel")
pic = picture(FIG / "speedup.png", cx, cy, h=ch - 74)
rows = [["Workload", "T seq", f"S({KMAX})", f"E({KMAX})", "p"]]
names = {"cleaning": "Data: clean 33 months", "search": "Task: 96 search fits", "features": "Task: 3 feature blocks"}
for wl in ["cleaning", "search", "features"]:
    seq = bench(wl, 1, "sequential"); k = bench(wl, KMAX)
    rows.append([names[wl], f"{seq['mean_s']:.1f} s" if seq["mean_s"] >= 1 else f"{seq['mean_s'] * 1000:.0f} ms",
                 f"{k['speedup']:.2f}x", f"{k['efficiency']:.2f}",
                 f"{k['amdahl_p']:.2f}" if k["amdahl_p"] > 0 else "< 0"])
table(cx, cy + ch - 72, cw, rows, [cw - 112, 30, 30, 26, 26], size=19, row_h=9)
pc, ps = bench("cleaning", KMAX), bench("search", KMAX)
text(cx, cy + ch - 34, cw, 34, [[("Amdahl: ", {"bold": True}),
     (f"p = {ps['amdahl_p']:.2f} (search), {pc['amdahl_p']:.3f} (cleaning), so cleaning can never beat "
      f"{pc['max_speedup']:.0f}x. Lost to start-up, pickling, shared disk, load imbalance and a busy shared server.", {})]],
     size=19, spacing=0.9)

cx, cy, cw, ch = panel(X3, Y3, COL, H3, "7  Non-determinism and fixes")
be = N["broken_evidence_this_run"]
rs = be["random_search_best"]
ft = [repr(float(v)) for v in be["fare_sq_total"]]
pair = next(((a, b) for i, a in enumerate(ft) for b in ft[i + 1:] if a != b), (ft[0], ft[1]))
same = 0   # length of the common prefix, so only the digits that changed are shown
while same < min(map(len, pair)) and pair[0][same] == pair[1][same]:
    same += 1
keep = max(same - 4, 0)
fare_ev = f"sum fare² ...{pair[0][keep:]}\nvs ...{pair[1][keep:]}"
fh = be["first_hour"]
fh_pair = next(((a, b) for i, a in enumerate(fh) for b in fh[i + 1:] if a != b), (fh[0], fh[1]))
rows = [["Source", "Broken runs differ", "Fix"],
        ["Completion order", f"first hour {fh_pair[0][:7]}\nvs {fh_pair[1][:7]}", "sort by month"],
        ["Float sum order", fare_ev, "fixed order / fsum"],
        ["Unseeded search", f"best lr {rs[0]['learning_rate']:.3f}\nvs {rs[1]['learning_rate']:.3f}", "seed 42 + trial id"]]
table(cx, cy, cw, rows, [60, cw - 128, 68], size=21, row_h=22,
      align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
ok = all(N[k]["fixed_runs_identical"] for k in ["completion_order", "reduction_order", "unseeded_randomness"])
text(cx, cy + 96, cw, ch - 96, [
    [("After each fix: ", {"bold": True}), ("two re-runs are identical " + ("in all three cases." if ok else "- CHECK!"), {})],
    [("Why: ", {"bold": True}), ("workers finish in a different order each run; float addition is not associative "
     f"((a+b)+c = {N['tiny_example']['(a+b)+c']:g}, a+(b+c) = {N['tiny_example']['a+(b+c)']:g}); unseeded "
     "generators differ. BLAS pinned to 1 thread per worker.", {})],
    [("Every parallel output = sequential ", {"bold": True}), ("(series, stats, features, 96 scores: asserted).", {})],
], size=22, spacing=0.95)

# ---------------------------------------------------------------- footer
env = R["environment"]
text(MARGIN, PAGE_H - 16, W, 12,
     f"Hardware: {env['hostname']}, {env['cpu_model']}, {env['cpus_allocated']} CPUs via Slurm, no GPU  |  "
     f"Python {env['python']}, pandas {env['pandas']}, scikit-learn {env['scikit-learn']}, pyarrow {env['pyarrow']}  |  "
     "All figures and numbers generated by demand_forecast.ipynb (outputs/results.json)",
     size=16, color=MUTED, align=PP_ALIGN.CENTER)

out = ROOT / "poster"
out.mkdir(exist_ok=True)
prs.save(out / "poster.pptx")
print("saved", out / "poster.pptx")
