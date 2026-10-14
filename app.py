"""
ibu — Insights on Breast-screening Users
FRE 6191 final project · Dash dashboard
Data: Breast Cancer Surveillance Consortium (BCSC) Risk Factors dataset,
      6.79 M screening mammograms, 2005-2017 (3 summarized CSV files in /data).
"""
import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, ctx, dcc, html
from dash.exceptions import PreventUpdate

import statemap

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

# ───────────────────────── palette (validated: dataviz skill) ─────────────────────────
PLUM, ROSE, ROSE_L, BLUSH, MIST = "#4A1D3F", "#C2456E", "#D9577A", "#F2C4D2", "#F7F4F6"
BLUE = "#5B6FB0"          # "comparison / earlier" colour; pairs with ROSE (CVD dE 10.6)
INK, MUTED, GRID, GRID2 = "#2E1027", "#6E4A64", "#E6DCE2", "#D3C6CE"
DENS_RAMP = ["#E58EA9", "#D0527A", "#A83A63", "#4A1D3F"]            # ordinal, 1 hue
AGE_RAMP = ["#E58EA9", "#D0527A", "#A83A63", "#7A2B5B", "#4A1D3F"]  # ordinal, 5 steps
BMI_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]             # ordinal blue
RACE = {1: ("White", "#5B6FB0"), 2: ("Black", "#D9962B"), 3: ("Asian / Pacific Isl.", "#2E9C8A"),
        5: ("Hispanic", "#8B6BB8"), 6: ("Other / mixed", "#C2456E")}
RACE_SHORT = {1: "W", 2: "B", 3: "A", 5: "H", 6: "O"}
RACE_X = {1: "White", 2: "Black", 3: "Asian/PI", 5: "Hispanic", 6: "Other"}
FONT = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"

AGE_LAB = {1: "18–29", 2: "30–34", 3: "35–39", 4: "40–44", 5: "45–49", 6: "50–54", 7: "55–59",
           8: "60–64", 9: "65–69", 10: "70–74", 11: "75–79", 12: "80–84", 13: "85+"}
BAND = {1: "Under 40", 2: "Under 40", 3: "Under 40", 4: "40–49", 5: "40–49", 6: "50–59", 7: "50–59",
        8: "60–69", 9: "60–69", 10: "70+", 11: "70+", 12: "70+", 13: "70+"}
BANDS = ["Under 40", "40–49", "50–59", "60–69", "70+"]
DENS_LAB = {1: "A · almost entirely fat", 2: "B · scattered density",
            3: "C · heterogeneously dense", 4: "D · extremely dense"}


# ───────────────────────── data ─────────────────────────
def load_cube() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA, "bcsc_risk_factors_summarized*.csv")))
    if not files:
        raise FileNotFoundError("Put the 3 BCSC csv files in ./data")
    d = pd.concat([pd.read_csv(f, dtype="int32") for f in files], ignore_index=True)
    d = d.rename(columns={"age_group_5_years": "age", "race_eth": "race", "first_degree_hx": "fh",
                          "BIRADS_breast_density": "dens", "current_hrt": "hrt", "menopaus": "meno",
                          "bmi_group": "bmi", "biophx": "bx", "breast_cancer_history": "bch"})
    d["race"] = d["race"].replace({4: 6})          # Native American (0.3 %) folded into Other
    keys = ["year", "age", "race", "fh", "dens", "hrt", "meno", "bmi", "bx", "bch"]
    c = d.groupby(keys, as_index=False)["count"].sum()   # drops menarche / first-birth (many unknowns)
    del d
    n = c["count"].astype("float64")
    c["n"] = n
    c["band"] = pd.Categorical(c["age"].map(BAND), categories=BANDS, ordered=True)

    def pair(num_mask, known_mask):
        return n * num_mask, n * known_mask

    c["sv"], c["svd"] = pair(c.bch == 1, c.bch != 9)
    c["dn"], c["dd"] = pair(c.dens.isin([3, 4]), c.dens != 9)
    c["hr"], c["hrd"] = pair(c.hrt == 1, c.hrt != 9)
    c["ob"], c["obd"] = pair(c.bmi.isin([3, 4]), c.bmi != 9)
    c["ob35"] = n * (c.bmi == 4)
    c["fh1"], c["fhd"] = pair(c.fh == 1, c.fh != 9)
    c["bx1"], c["bxd"] = pair(c.bx == 1, c.bx != 9)
    c["a65"] = n * (c.age >= 9)
    c["nw"], c["rd"] = pair(c.race.isin([2, 3, 5, 6]), c.race != 9)
    return c


C = load_cube()
Y0, Y1 = int(C.year.min()), int(C.year.max())

TRAITS = {  # label -> (numerator col, denominator col)
    "Hormone therapy (HRT)": ("hr", "hrd"),
    "Obese (BMI ≥ 30)": ("ob", "obd"),
    "Severely obese (BMI ≥ 35)": ("ob35", "obd"),
    "Dense breasts (C/D)": ("dn", "dd"),
    "Family history": ("fh1", "fhd"),
    "Prior biopsy": ("bx1", "bxd"),
    "Prior breast cancer": ("sv", "svd"),
    "Age 65+": ("a65", "n"),
    "Non-White (race known)": ("nw", "rd"),
}


def sub(yrs):
    return C[(C.year >= yrs[0]) & (C.year <= yrs[1])]


def pct(num, den):
    return 100.0 * num / den if den else np.nan


def rate_by(df, by, num, den):
    g = df.groupby(by, observed=True)[[num, den]].sum()
    return 100.0 * g[num] / g[den]


def std_by(df, grp, num, den):
    """Direct age-standardised % per group (standard = age mix of df)."""
    t = df.groupby([grp, "age"], observed=True)[[num, den]].sum()
    r = (t[num] / t[den]).unstack(0)
    w = df.groupby("age")[den].sum()
    w = (w / w.sum()).reindex(r.index).values[:, None] * r.notna().values
    return pd.Series((r.fillna(0).values * w).sum(0) / w.sum(0) * 100, index=r.columns)


# ───────────────────────── U.S. state data (separate source: NCI / CDC State Cancer Profiles) ─────────────────────────
STATE = pd.read_csv(os.path.join(DATA, "us_state_context.csv")).sort_values("state").reset_index(drop=True)
STATE["mi"] = 100 * STATE.mort / STATE.inc            # deaths per 100 new cases
SHAPES, LABELS = statemap.load_state_shapes()
# key -> (dropdown label, short label, number format)
GEO = {
    "inc": ("New cases per 100,000 women", "new-case rate", "{:.0f}"),
    "mort": ("Deaths per 100,000 women", "death rate", "{:.1f}"),
    "mi": ("Deaths per 100 new cases (how often it is fatal)", "deaths per 100 cases", "{:.1f}"),
    "obese": ("Adults who are obese (%)", "obesity rate", "{:.1f}%"),
    "smoke": ("Adults who smoke (%)", "smoking rate", "{:.1f}%"),
    "inactive": ("Adults with no physical activity (%)", "inactivity rate", "{:.1f}%"),
    "fruit": ("Adults who eat fruit daily (%)", "daily fruit eating", "{:.1f}%"),
    "veg": ("Adults who eat vegetables daily (%)", "daily vegetable eating", "{:.1f}%"),
    "unins": ("Adults under 65 without health insurance (%)", "uninsured rate", "{:.1f}%"),
    "pov": ("People below 150% of the poverty line (%)", "poverty rate", "{:.1f}%"),
    "income": ("Median family income ($)", "income", "${:,.0f}"),
    "bach": ("Adults with a bachelor’s degree (%)", "college-degree share", "{:.1f}%"),
}
MAP_RAMP = ["#F6DCE5", "#E58EA9", "#D0527A", "#A83A63", "#7A2B5B", "#4A1D3F"]   # one hue, light → dark


def _hex(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def ramp(t):
    t = float(np.clip(t, 0, 1)) * (len(MAP_RAMP) - 1)
    i = min(int(t), len(MAP_RAMP) - 2)
    r = _hex(MAP_RAMP[i]) * (1 - (t - i)) + _hex(MAP_RAMP[i + 1]) * (t - i)
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in r)


# ───────────────────────── figure helpers ─────────────────────────
def base(fig, h=380, l=60, r=30, t=56, b=None, legend=True):
    if b is None:
        b = 100 if legend else 50
    fig.update_layout(
        height=h, margin=dict(l=l, r=r, t=t, b=b), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=13, color=INK),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=13, color=INK)),
        legend=dict(orientation="h", y=-0.3, yanchor="top", x=0, font=dict(size=12)), showlegend=legend,
        title=dict(x=0, xanchor="left", y=0.97, font=dict(size=16, color=INK)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID2, ticks="outside", tickcolor=GRID2)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig


def title(fig, text):
    fig.update_layout(title_text=text)
    return fig


def empty(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=15, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return base(fig, 300, legend=False)


def dumbbell(labels, a, b, an, bn, ac, bc, h=420, l=190, xtitle="% of exams", delta_unit=" pp"):
    fig = go.Figure()
    xs, ys = [], []
    for lab, x0, x1 in zip(labels, a, b):
        xs += [x0, x1, None]
        ys += [lab, lab, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=GRID2, width=3),
                             hoverinfo="skip", showlegend=False))
    for name, vals, col in ((an, a, ac), (bn, b, bc)):
        fig.add_trace(go.Scatter(x=vals, y=labels, mode="markers", name=name,
                                 marker=dict(color=col, size=11, line=dict(color="white", width=2)),
                                 hovertemplate="%{y}<br>" + name + ": <b>%{x:.1f}%</b><extra></extra>"))
    top = [max(x0, x1) for x0, x1 in zip(a, b)]
    fig.add_trace(go.Scatter(x=top, y=labels, mode="text", showlegend=False, hoverinfo="skip",
                             text=[f"  {y - x:+.1f}{delta_unit}" for x, y in zip(a, b)],
                             textposition="middle right", textfont=dict(size=12, color=INK)))
    base(fig, h, l=l, r=70)
    fig.update_yaxes(categoryorder="array", categoryarray=labels, autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text=xtitle, rangemode="tozero", showgrid=True, gridcolor=GRID)
    return fig


def heat(z, x, y, scale, zmid=None, zmin=None, zmax=None, fmt="{:.0f}", cut=0.55, hover="", h=420,
         cbar="", xtitle="", ytitle=""):
    z = np.asarray(z, dtype=float)
    lo = np.nanmin(z) if zmin is None else zmin
    hi = np.nanmax(z) if zmax is None else zmax
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=scale, zmid=zmid, zmin=lo, zmax=hi,
                               xgap=2, ygap=2, hovertemplate=hover + "<extra></extra>",
                               colorbar=dict(title=dict(text=cbar, side="right"), thickness=12, len=0.9)))
    notes = []
    for i, yy in enumerate(y):
        for j, xx in enumerate(x):
            v = z[i, j]
            if np.isnan(v):
                continue
            f = (v - lo) / (hi - lo) if hi > lo else 0.5
            if zmid is not None:
                f = abs(v - zmid) / max(abs(hi - zmid), abs(zmid - lo), 1e-9) * 0.5 + 0.5
            notes.append(dict(x=j, y=i, text=fmt(v) if callable(fmt) else fmt.format(v), showarrow=False,
                              font=dict(size=11, color="white" if f > cut else INK)))
    base(fig, h, l=70, r=20, legend=False)
    fig.update_layout(annotations=notes)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(type="category", title_text=xtitle)
    fig.update_yaxes(title_text=ytitle)
    return fig


# ───────────────────────── layout helpers ─────────────────────────
CFG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def graph(gid, h=None):
    return dcc.Graph(id=gid, config=CFG, style={"height": f"{h}px"} if h else {})


def card(head, sub, body, wide=False, note=None):
    kids = [html.H3(head), html.P(sub, className="card-sub"), body]
    if note:
        kids.append(html.P(note, className="card-note"))
    return html.Div(kids, className="card reveal wide" if wide else "card reveal")


def chapter(cid, num, kicker, head, lede, cards, sowhat, extra=None):
    kids = [html.Div([html.Span(num, className="chnum"), html.Span(kicker, className="kicker")],
                     className="chhead"),
            html.H2(head), html.P(lede, className="lede")]
    if extra is not None:
        kids.append(extra)
    kids.append(html.Div(cards, className="grid"))
    kids.append(html.Div([html.Span("Why it matters for a screening provider", className="sw-label"),
                          html.P(sowhat)], className="sowhat reveal"))
    return html.Section(kids, id=cid, className="chapter")


def tile(label, value, sub="", cls=""):
    return html.Div([html.Div(label, className="t-label"), html.Div(value, className="t-value"),
                     html.Div(sub, className="t-sub")], className=f"tile {cls}")


# ───────────────────────── app ─────────────────────────
app = Dash(__name__, title="ibu · who is in the mammogram chair?", update_title=None,
           suppress_callback_exceptions=False)
server = app.server

NAV = [("ch1", "Prevention"), ("geo", "Geography"), ("ch2", "Detection"), ("ch3", "Screening"),
       ("ch4", "Risk"), ("ch5", "Survivors"), ("ch6", "Business"), ("ch7", "Takeaways"), ("contact", "Contact")]
LINKEDIN, GITHUB = "https://www.linkedin.com/in/azfina", "https://github.com/azfina"


def icon_link(kind, href, label, text=None):
    kids = [html.Span(className=f"ico {kind}")] + ([html.Span(text)] if text else [])
    return html.A(kids, href=href, target="_blank", rel="noopener noreferrer", className="icon-link",
                  title=label, **{"aria-label": label})


header = html.Header([
    html.Div([
        html.A([html.Img(src="/assets/logo.svg", className="logo", alt=""),
                html.Span("ibu", className="brand")], href="#landing", className="brand-link"),
        html.Nav([html.A(t, href=f"#{i}") for i, t in NAV], className="nav"),
        html.Div([icon_link("li", LINKEDIN, "LinkedIn"), icon_link("gh", GITHUB, "GitHub")], className="hicons"),
    ], className="hbar"),
    html.Div([
        html.Div([html.Span("Years shown", className="f-label"),
                  html.Span("drag the handles · charts that use the screening data respond", className="f-hint")],
                 className="f-text"),
        html.Div(dcc.RangeSlider(id="years", min=Y0, max=Y1, step=1, value=[Y0, Y1], allowCross=False,
                                 marks={y: str(y) for y in range(Y0, Y1 + 1, 2)},
                                 tooltip={"placement": "bottom", "always_visible": False}),
                 className="f-slider"),
    ], className="fbar"),
], id="header", className="header")


def stat(value, label, count=None, suffix="", src=""):
    num = html.Span(value, className="big-num", **({"data-count": str(count), "data-suffix": suffix} if count else {}))
    return html.Div([num, html.Div(label, className="stat-label"), html.Div(src, className="stat-src")],
                    className="stat reveal")


def surv_bar(label, pct_, cls):
    return html.Div([html.Div([html.Span(label), html.B(f"{pct_}%")], className="sb-top"),
                     html.Div(html.Div(className=f"sb-fill {cls}", style={"--w": f"{pct_}%"}), className="sb-track")],
                    className="sb-row")


landing = html.Section([
    html.Div([html.Span(className=f"rp rp{i}") for i in range(1, 5)], className="ripples", **{"aria-hidden": "true"}),
    html.Div([
        html.Div([
            html.Div("Breast Cancer Awareness Month · ibu = “mother” in Indonesian", className="eyebrow rise r1"),
            html.H1("Found early, she goes home.", className="rise r2"),
            html.P("When breast cancer is found while it is still in the breast, more than 99 of every 100 women are "
                   "alive five years later. Found after it has spread, it is 32 of 100. The difference is often one "
                   "appointment, and behind every number on this page is somebody’s mother.", className="lede big rise r3"),
            html.Div([html.A("Begin the story", href="#story", className="cta"),
                      html.A("Jump to the map", href="#geo", className="cta ghost")], className="cta-row rise r4"),
        ], className="land-text"),
        html.Div([
            html.Div([html.Div("Five-year survival by stage at diagnosis", className="sb-head"),
                      surv_bar("Found in the breast (localized)", 99, "a"),
                      surv_bar("Spread to nearby nodes (regional)", 87, "b"),
                      surv_bar("Spread to other organs (distant)", 32, "c"),
                      html.Div("American Cancer Society, women diagnosed 2015–2021", className="stat-src")],
                     className="surv-card reveal"),
            html.Div([html.Span(className="dot" + (" hot" if i == 7 else "")) for i in range(8)],
                     className="dots reveal"),
            html.Div([html.B("1 in 8"), " women in the U.S. will be diagnosed in her lifetime."], className="dots-cap"),
        ], className="land-viz"),
    ], className="land-in"),
    html.Div([
        stat("321,910", "new invasive breast cancers expected in U.S. women in 2026", 321910, "", "American Cancer Society"),
        stat("4,000,000+", "breast-cancer survivors living in the U.S. today (about)", 4000000, "+", "American Cancer Society"),
        stat("2,400,000", "women diagnosed worldwide in 2024; most common cancer in women in 164 of 186 countries", 2400000, "",
             "World Health Organization"),
    ], className="stats"),
    html.Div([html.Span("For every ibu — and the families who walked with her."), html.A("↓", href="#story", className="scroll-cue",
              **{"aria-label": "Scroll to the story"})], className="dedication"),
], id="landing", className="landing")

hero = html.Section([
    html.Div([
        html.Div([
            html.Div("ibu = “mother” in Indonesian", className="eyebrow"),
            html.H1(["Who is in the mammogram chair —", html.Br(), "and how is she changing?"]),
            html.P("A story told through 6.79 million U.S. screening mammograms (2005–2017): what changed "
                   "before the scan, who gets scanned, what the scan can see, and the growing group of women "
                   "who come back after surviving breast cancer.", className="lede"),
            html.Details([
                html.Summary("How to use this dashboard"),
                html.Ul([
                    html.Li([html.B("Scroll"), " — the page is a story in 8 chapters; use the top menu to jump."]),
                    html.Li([html.B("Years slider"), " (top bar) — narrows every chart to those years."]),
                    html.Li([html.B("Hover"), " any mark for exact numbers and sample sizes."]),
                    html.Li([html.B("Toggles & dropdowns"), " inside cards switch the measure (e.g. % vs. gap)."]),
                    html.Li([html.B("Map"), " (chapter 2) — pick a measure, then click a state to see its numbers."]),
                    html.Li([html.B("Treemap"), " — click a tile to zoom in; click the top bar to zoom out."]),
                    html.Li([html.B("Persona builder"), " (chapter 7) — build a patient profile and see her segment."]),
                    html.Li([html.B("Legend"), " — click a name to hide/show a series; double-click to isolate."]),
                    html.Li([html.B("Camera icon"), " on any chart saves it as a PNG."]),
                ], className="howto"),
            ], className="details", open=False),
        ], className="hero-text"),
        html.Div(id="kpis", className="kpis"),
    ], className="hero-in"),
], id="story", className="hero")

ch1 = chapter(
    "ch1", "1", "Prevention", "The risk profile that can change is changing",
    "Weight and hormone therapy are two of the few breast-cancer risk factors a woman can influence. "
    "Across 13 years of screened women, one moved in the wrong direction and one in the right one.",
    [card("What shifted most between the first and last year selected",
          "Each row is a trait; the dots show the share of exams with that trait in the first (blue) and last (rose) "
          "year of your selection. Sorted by size of change.", graph("g_change", 430), wide=True,
          note="Shares use exams where the trait is known. Prior breast cancer is shown in chapter 6."),
     card("Hormone therapy fell in every age band",
          "Current HRT users as % of post-menopausal exams, by age band.", graph("g_hrt", 380)),
     card("Obesity rose for every group — but not equally",
          "Share of exams with BMI ≥ 30, by race / ethnicity (where BMI is known).", graph("g_obrace", 380))],
    "Prevention messaging is not generic: HRT counselling has already done its work, while weight-related risk is "
    "rising fastest in specific groups — a clear target for outreach and for partnerships with primary care.")

ch2 = chapter(
    "ch2", "3", "Detection", "Who actually walks in for screening?",
    "Detection starts with showing up. Screening exams are concentrated in midlife, and the people showing up are "
    "becoming older and more diverse.",
    [card("Screening is a midlife service",
          "Share of exams by age group. Rose bars are the 40–74 core screening ages.", graph("g_age", 380)),
     card("The patient mix is diversifying",
          "Share of exams by race / ethnicity (unknown excluded), stacked to 100 %.", graph("g_race", 380))],
    "Staffing, language support and appointment times should follow the patient mix of the next five years, "
    "not the last five.")

ch3 = chapter(
    "ch3", "4", "Screening", "What the scan can see: breast density",
    "Dense breast tissue appears white on a mammogram — the same colour as many tumours — so it can hide cancers and "
    "is itself a risk marker. Almost half of all exams are of dense breasts.",
    [card("Density fades with age",
          "BI-RADS density category, share of exams with known density, by age group.", graph("g_dens", 400)),
     card("Who has the densest breasts?",
          "Age matters, so compare women of the same age. Switch between raw % and the gap to all women of that age.",
          html.Div([dcc.RadioItems(id="heat_mode",
                                   options=[{"label": " % with dense breasts", "value": "pct"},
                                            {"label": " Gap vs. all women of same age (points)", "value": "gap"}],
                                   value="pct", inline=True, className="radio"),
                    graph("g_heat", 400)])),
     card("Heavier women tend to have less dense breasts",
          "Age-standardised % dense by BMI group, so the comparison is not just about age.", graph("g_bmi", 340),
          wide=True)],
    "Since September 2024 U.S. facilities must tell every woman her density (FDA rule — context outside this "
    "dataset). Many women told ‘dense’ will ask about 3D tomosynthesis, ultrasound or MRI. The segment sizes here show how "
    "big that conversation is.")

ch4 = chapter(
    "ch4", "5", "Risk markers", "Who has already been through it?",
    "This dataset holds only women who were screened — it does not record new diagnoses or deaths. What it does "
    "record is whether a woman has had breast cancer before, so we use that to see which markers travel with a "
    "prior diagnosis. These are associations, not causes.",
    [card("Risk markers, adjusted for age",
          "Age-standardised % of exams from women with a prior breast cancer, by marker. "
          "† marks things that can be a consequence of a diagnosis (HRT stopped, surgical menopause).",
          graph("g_lolli", 560), wide=True,
          note="Dashed line = overall rate. Rows are exams with that marker known. Prior biopsy is left out: women with a "
              "biopsy are over 11× as likely to be survivors, but a biopsy is part of the diagnosis, not a risk marker."),
     card("A family history lifts the risk at every screening age",
          "Prior breast cancer % by age group, women with vs. without a mother/sister/daughter with breast cancer.",
          graph("g_famage", 380), wide=True)],
    "Family history is the one marker a provider can ask about at the front desk; it is the simplest trigger for "
    "an earlier-start or supplemental-imaging conversation.")

ch5 = chapter(
    "ch5", "6", "Survivors", "The fastest-growing group in the chair",
    "Women with a prior breast cancer diagnosis are a rising share of every year’s exams. Part of that is simply "
    "an older screening population — but not all of it.",
    [card("How much of the rise is just ageing?",
          "Survivors as % of exams: as observed vs. if the age mix had stayed fixed at the average of your selection.",
          graph("g_surv", 380)),
     card("Where the survivors are",
          "Survivor share by age group and year. Switch to see counts of exams instead of shares.",
          html.Div([dcc.RadioItems(id="surv_mode",
                                   options=[{"label": " % of exams", "value": "pct"},
                                            {"label": " Number of exams", "value": "n"}],
                                   value="pct", inline=True, className="radio"),
                    graph("g_survheat", 400)])),
     card("How survivors differ from other women",
          "Share of each group with the trait. Age-adjusted removes the fact that survivors are older.",
          html.Div([dcc.RadioItems(id="prof_mode",
                                   options=[{"label": " Age-adjusted", "value": "adj"},
                                            {"label": " Raw", "value": "raw"}],
                                   value="adj", inline=True, className="radio"),
                    graph("g_profile", 380)]), wide=True,
          note="Prior biopsy is left out: it is part of the diagnosis, not a difference. Lower HRT among survivors is largely medical advice after diagnosis, not a risk factor.")],
    "Survivors come back every year and need follow-up care, not just screening. "
    "A dedicated survivorship track (scheduling, imaging pathway, support) serves a segment that is already growing.")

ch6 = chapter(
    "ch6", "7", "Business", "Know your patient: segments and personas",
    "Put the previous chapters together. Which segments are biggest, which carry the most survivors, and what does "
    "your own persona look like?",
    [card("Where the volume sits",
          "Exam volume by age band and density (known density only). Tile size = exams; colour = chosen measure. "
          "Click a tile to zoom.",
          html.Div([dcc.Dropdown(id="tree_metric",
                                 options=[{"label": k, "value": k} for k in TRAITS if k != "Age 65+"],
                                 value="Prior breast cancer", clearable=False, className="dd"),
                    graph("g_tree", 440)])),
     card("Which age × ethnicity segments are large and distinctive?",
          "Bubble = segment (letter = race / ethnicity). Size = exams, colour = age band. "
          "Segments under 5,000 exams hidden.", graph("g_bubble", 440)),
     card("Persona builder",
          "Choose a profile. We show how many exams share it, what is different about her segment and how its share "
          "of all exams has moved.",
          html.Div([
              html.Div([
                  html.Div([html.Label("Age band"),
                            dcc.Dropdown(id="p_age", options=["All"] + BANDS, value="60–69", clearable=False)]),
                  html.Div([html.Label("Race / ethnicity"),
                            dcc.Dropdown(id="p_race", options=["All"] + [v[0] for v in RACE.values()],
                                         value="All", clearable=False)]),
                  html.Div([html.Label("Breast density"),
                            dcc.Dropdown(id="p_dens", options=["All", "Dense (C/D)", "Not dense (A/B)"],
                                         value="Dense (C/D)", clearable=False)]),
                  html.Div([html.Label("BMI"),
                            dcc.Dropdown(id="p_bmi", options=["All", "BMI < 30", "BMI ≥ 30"],
                                         value="All", clearable=False)]),
                  html.Div([html.Label("Family history"),
                            dcc.Dropdown(id="p_fh", options=["All", "Yes", "No"], value="All", clearable=False)]),
              ], className="persona-controls"),
              html.Div(id="persona_tiles", className="persona-tiles"),
              graph("g_persona", 260)]), wide=True)],
    "The same data becomes a targeting tool: size a segment, see whether it is growing, and compare its survivor "
    "and density profile to the average before spending a marketing dollar.")

geo = chapter(
    "geo", "2", "Geography", "Same disease, different states",
    "Breast cancer is not the same everywhere. Food, exercise, smoking, income and health insurance all differ by state, "
    "and so does what is found and what is lost. This chapter uses a second, public data source (U.S. state cancer "
    "statistics, 2015) and does not change with the years slider. Pick a measure, then click a state.",
    [card("The map of breast cancer in America",
          "Darker = higher. Switch the measure to compare cancer rates with lifestyle and income. Click any state to "
          "select it.",
          html.Div([dcc.Dropdown(id="geo_metric", options=[{"label": v[0], "value": k} for k, v in GEO.items()],
                                 value="inc", clearable=False, className="dd wide-dd"),
                    graph("g_map", 640)]), wide=True,
          note="Rates are age-adjusted. A high new-case rate can mean more cancer or more screening: states that screen "
               "more find more. Deaths per 100 new cases compares how often a diagnosis is fatal, so it is a better "
               "signal of access to care."),
     card("Zoom into one state",
          "Chosen by clicking the map or a dot below. Rank 1 = highest of 51 (50 states and D.C.).",
          html.Div([dcc.Dropdown(id="geo_state", options=list(STATE.state), value="New York", clearable=False,
                                 className="dd"),
                    html.Div(id="geo_tiles", className="persona-tiles")]), wide=True),
     card("Do lifestyle and income line up with breast cancer?",
          "Each dot is a state. Pick what to put on each axis. A line is fitted to see the direction.",
          html.Div([html.Div([html.Div([html.Label("Across (x)"),
                                        dcc.Dropdown(id="geo_x", options=[{"label": v[0], "value": k} for k, v in GEO.items()
                                                                          if k not in ("inc", "mort", "mi")],
                                                     value="income", clearable=False)]),
                              html.Div([html.Label("Up (y)"),
                                        dcc.Dropdown(id="geo_y", options=[{"label": GEO[k][0], "value": k}
                                                                          for k in ("inc", "mort", "mi")],
                                                     value="inc", clearable=False)])], className="duo"),
                    graph("g_geo_sc", 440)]), wide=True,
          note="States, not people: a pattern across states does not prove what happens to any one woman. "
               "Many things travel together (income, insurance, screening, age mix).")],
    "Marketing is local. Where income and insurance are high, women are already finding cancer, so the message is "
    "‘keep your yearly appointment’. Where poverty is high and a diagnosis is more often fatal, the barrier is access "
    "and cost: mobile units, sliding-scale pricing, community partners and a call to a trusted person. One national "
    "campaign would miss both.")

ch7 = html.Section([
    html.Div([html.Span("8", className="chnum"), html.Span("Takeaways", className="kicker")], className="chhead"),
    html.H2("What ibu found"),
    html.Div(id="takeaways", className="take-grid"),
], id="ch7", className="chapter")

contact = html.Section([
    html.Div([
        html.Div([html.Div("Contact", className="kicker light"), html.H2("Built by Azfina Putri Anindita"),
                  html.P("NYU · FRE 6191 Data Visualization. Questions, ideas or feedback on ibu are welcome.")],
                 className="ct-text"),
        html.Div([icon_link("li", LINKEDIN, "LinkedIn", "LinkedIn · /in/azfina"),
                  icon_link("gh", GITHUB, "GitHub", "GitHub · @azfina")], className="ct-links"),
    ], className="ct-in"),
], id="contact", className="contact")

about = html.Section([
    html.H2("About the data"),
    html.Details([
        html.Summary("More information about this data"),
        html.Div([
            html.P([html.B("Source. "), "Breast Cancer Surveillance Consortium (BCSC) Risk Factors Dataset, "
                    "summarized version 09/2020 — 6,788,436 screening and diagnostic mammograms from BCSC registries, "
                    "Jan 2005 – Dec 2017. ",
                    html.A("bcsc-research.org/datasets/rf", href="https://www.bcsc-research.org/datasets/rf",
                           target="_blank", rel="noopener"),
                    ". Supported by the National Cancer Institute (P01CA154292). Downloaded October 2026."]),
            html.P([html.B("Unit. "), "One row = a combination of risk factors; ",
                    html.Code("count"), " = number of exams with that combination (one exam per woman per calendar "
                    "year). All rates are weighted by count. We talk about exams, not women: the same woman can "
                    "appear in several years."]),
            html.P([html.B("Files. "), "The 3 CSVs are one dataset cut by year (2005–09, 2009–13, 2013–17) and are "
                    "combined here."]),
            html.P([html.B("Unknowns. "), "Each measure excludes exams where that measure is unknown "
                    "(race 5 %, density 11 %, BMI 36 %, HRT 12 %, prior cancer 16 %). Age at menarche (61 % unknown) "
                    "and age at first birth (33 % unknown) are not used. Native American (0.3 %) is folded into "
                    "‘Other / mixed’."]),
            html.P([html.B("Age adjustment. "), "‘Age-standardised’ = rates computed within each age group and "
                    "re-weighted to the age mix of the selected exams, so groups are compared at equal age."]),
            html.P([html.B("State map (chapter 2). "), "A separate source: NCI / CDC State Cancer Profiles — female "
                    "breast-cancer new-case and death rates (age-adjusted, per 100,000, 2015), 2016 BRFSS lifestyle "
                    "survey, 2012–16 American Community Survey (insurance, poverty, income, education), compiled in the "
                    "open repository ",
                    html.A("sergeevaleeza/cancer-data-visualization-america", href="https://github.com/sergeevaleeza/cancer-data-visualization-america",
                           target="_blank", rel="noopener"),
                    " (MIT). Boundaries: U.S. Census Bureau 2010, 1:20 million. Alaska and Hawaii are drawn as insets. "
                    "Landing-page figures: American Cancer Society (Cancer Statistics 2026) and WHO (2024)."]),
            html.P([html.B("Limits. "), "Associations, not causes. ‘Prior breast cancer’ is history, not new "
                    "diagnosis. Volume falls after 2010 mostly because fewer facilities report to BCSC, so we show "
                    "shares, not totals. Small age groups (under 40) are noisy — hover shows n."]),
        ], className="about-body"),
    ], className="details"),
    html.P("FRE 6191 · Data visualization · Built with Dash and Plotly", className="footer"),
], className="about")

app.layout = html.Div([header, landing, hero, ch1, geo, ch2, ch3, ch4, ch5, ch6, ch7, about, contact], className="page")


# ───────────────────────── callbacks ─────────────────────────
@app.callback(Output("kpis", "children"), Input("years", "value"))
def kpis(yrs):
    df = sub(yrs)
    last = df[df.year == yrs[1]]
    return [
        tile("Mammograms in view", f"{df.n.sum() / 1e6:.2f} M", f"{yrs[0]}–{yrs[1]}"),
        tile("From survivors", f"{pct(last.sv.sum(), last.svd.sum()):.1f}%", f"of exams in {yrs[1]}", "rose"),
        tile("Dense breasts", f"{pct(df.dn.sum(), df.dd.sum()):.0f}%", "of exams (density known)"),
        tile("Aged 65+", f"{pct(last.a65.sum(), last.n.sum()):.0f}%", f"of exams in {yrs[1]}"),
    ]


# 1a
@app.callback(Output("g_change", "figure"), Input("years", "value"))
def g_change(yrs):
    if yrs[0] == yrs[1]:
        return empty("Select two different years to see change.")
    d0, d1 = C[C.year == yrs[0]], C[C.year == yrs[1]]
    rows = []
    for lab, (num, den) in TRAITS.items():
        if lab == "Prior breast cancer":
            continue
        rows.append((lab, pct(d0[num].sum(), d0[den].sum()), pct(d1[num].sum(), d1[den].sum())))
    rows.sort(key=lambda r: -(r[2] - r[1]))
    lab, a, b = zip(*rows)
    fig = dumbbell(list(lab), list(a), list(b), str(yrs[0]), str(yrs[1]), BLUE, ROSE)
    big = rows[0]
    return title(fig, f"{big[0]} grew most: {big[1]:.1f}% → {big[2]:.1f}%")


# 1b
@app.callback(Output("g_hrt", "figure"), Input("years", "value"))
def g_hrt(yrs):
    df = sub(yrs)
    df = df[(df.meno == 2) & (df.band.isin(["50–59", "60–69", "70+"]))]
    cols = {"50–59": AGE_RAMP[1], "60–69": AGE_RAMP[2], "70+": AGE_RAMP[4]}
    fig = go.Figure()
    s = None
    for bnd in cols:
        s = rate_by(df[df.band == bnd], "year", "hr", "hrd")
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=bnd,
                                 line=dict(color=cols[bnd], width=2.5),
                                 marker=dict(size=8, line=dict(color="white", width=2)),
                                 hovertemplate=f"Age {bnd}<br>%{{x}}: <b>%{{y:.1f}}%</b><extra></extra>"))
        fig.add_annotation(x=s.index[-1], y=s.values[-1], text=f" {bnd}", showarrow=False, xanchor="left",
                           yshift={"50–59": 0, "60–69": 7, "70+": -7}[bnd], font=dict(size=12, color=INK))
    base(fig, 380, r=60)
    fig.update_yaxes(title_text="% on hormone therapy", rangemode="tozero")
    fig.update_xaxes(dtick=2, range=[yrs[0] - 0.3, yrs[1] + 0.3])
    hi = rate_by(df[df.band == "50–59"], "year", "hr", "hrd")
    return title(fig, f"Age 50–59 HRT use: {hi.iloc[0]:.1f}% → {hi.iloc[-1]:.1f}%")


# 1c
@app.callback(Output("g_obrace", "figure"), Input("years", "value"))
def g_obrace(yrs):
    df = sub(yrs)
    fig = go.Figure()
    last = {}
    for code, (nm, col) in RACE.items():
        s = rate_by(df[df.race == code], "year", "ob", "obd")
        last[nm] = s.iloc[-1]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=nm,
                                 line=dict(color=col, width=2.5),
                                 marker=dict(size=7, line=dict(color="white", width=2)),
                                 hovertemplate=f"{nm}<br>%{{x}}: <b>%{{y:.1f}}%</b><extra></extra>"))
    base(fig, 380, r=30)
    fig.update_yaxes(title_text="% with BMI ≥ 30", rangemode="tozero")
    fig.update_xaxes(dtick=2, range=[yrs[0] - 0.3, yrs[1] + 0.3])
    top = max(last, key=last.get)
    return title(fig, f"Highest in {yrs[1]}: {top} women, {last[top]:.0f}% with BMI ≥ 30")


# 2a
@app.callback(Output("g_age", "figure"), Input("years", "value"))
def g_age(yrs):
    df = sub(yrs)
    s = df.groupby("age").n.sum()
    s = 100 * s / s.sum()
    core = s.loc[4:10].sum()
    cols = [ROSE if 4 <= a <= 10 else GRID2 for a in s.index]
    fig = go.Figure(go.Bar(x=[AGE_LAB[a] for a in s.index], y=s.values, marker=dict(color=cols,
                           line=dict(color="white", width=2)),
                           hovertemplate="Age %{x}<br><b>%{y:.1f}%</b> of exams<extra></extra>"))
    base(fig, 380, legend=False)
    fig.update_yaxes(title_text="% of exams", rangemode="tozero")
    fig.update_xaxes(title_text="Age group (years)")
    return title(fig, f"{core:.0f}% of exams are for women aged 40–74")


# 2b
@app.callback(Output("g_race", "figure"), Input("years", "value"))
def g_race(yrs):
    df = sub(yrs)
    df = df[df.race != 9]
    t = df.groupby(["year", "race"]).n.sum().unstack()
    t = 100 * t.div(t.sum(1), axis=0)
    fig = go.Figure()
    for code, (nm, col) in RACE.items():
        fig.add_trace(go.Scatter(x=t.index, y=t[code], name=nm, mode="lines", stackgroup="one",
                                 line=dict(width=1, color="white"), fillcolor=col,
                                 hovertemplate=f"{nm}<br>%{{x}}: <b>%{{y:.1f}}%</b><extra></extra>"))
    cum = 0
    for code, (nm, col) in RACE.items():
        v = t[code].iloc[-1]
        fig.add_annotation(x=t.index[-1], y=cum + v / 2, text=f"<span style='color:{col}'>■</span> {nm} {v:.0f}%",
                           showarrow=False, xanchor="left", xshift=6, font=dict(size=12, color=INK))
        cum += v
    base(fig, 380, r=170, legend=False)
    fig.update_yaxes(title_text="% of exams", range=[0, 100], ticksuffix="")
    fig.update_xaxes(dtick=2, range=[yrs[0], yrs[1]], tickangle=0)
    nonw = 100 - t[1]
    return title(fig, f"Non-White share of exams: {nonw.iloc[0]:.0f}% → {nonw.iloc[-1]:.0f}%")


# 3a
@app.callback(Output("g_dens", "figure"), Input("years", "value"))
def g_dens(yrs):
    df = sub(yrs)
    df = df[df.dens != 9]
    t = df.groupby(["age", "dens"]).n.sum().unstack()
    t = 100 * t.div(t.sum(1), axis=0)
    fig = go.Figure()
    for d in (1, 2, 3, 4):
        fig.add_trace(go.Bar(x=[AGE_LAB[a] for a in t.index], y=t[d], name=DENS_LAB[d],
                             marker=dict(color=DENS_RAMP[d - 1], line=dict(color="white", width=2)),
                             hovertemplate=f"{DENS_LAB[d]}<br>Age %{{x}}: <b>%{{y:.1f}}%</b><extra></extra>"))
    fig.update_layout(barmode="stack")
    base(fig, 400)
    fig.update_yaxes(title_text="% of exams", range=[0, 100])
    fig.update_xaxes(title_text="Age group (years)")
    dn = t[[3, 4]].sum(1)
    return title(fig, f"Dense (C+D): {dn.iloc[0]:.0f}% of exams under 30 → {dn.iloc[-1]:.0f}% at 85+")


# 3b
@app.callback(Output("g_heat", "figure"), Input("years", "value"), Input("heat_mode", "value"))
def g_heat(yrs, mode):
    df = sub(yrs)
    df = df[(df.dens != 9) & (df.race != 9) & (df.age.between(4, 12))]
    g = df.groupby(["age", "race"])[["dn", "dd"]].sum()
    z = (100 * g.dn / g.dd).unstack()[list(RACE)]
    overall = 100 * df.groupby("age").dn.sum() / df.groupby("age").dd.sum()
    xs = [RACE_X[c] for c in z.columns]
    ys = [AGE_LAB[a] for a in z.index]
    if mode == "gap":
        zz = z.sub(overall, axis=0).values
        lim = np.nanmax(np.abs(zz))
        fig = heat(zz, xs, ys, [[0, BLUE], [0.5, "#F0EEF0"], [1, ROSE]], zmid=0, zmin=-lim, zmax=lim,
                   fmt="{:+.0f}", hover="Age %{y} · %{x}<br>Gap vs same-age average: <b>%{z:+.1f} pts</b>",
                   cbar="points", cut=0.7, xtitle="", ytitle="Age")
        asian = (z[3] - overall).mean()
        ttl = f"Asian / PI women: on average {asian:+.0f} points denser than same-age women"
    else:
        fig = heat(z.values, xs, ys, [[0, "#FBEAF0"], [0.5, "#E07C9B"], [1, PLUM]], hover="Age %{y} · %{x}<br>"
                   "<b>%{z:.1f}%</b> dense", cbar="% dense", cut=0.5, xtitle="", ytitle="Age")
        ia, ir = np.unravel_index(np.nanargmax(z.values), z.values.shape)
        ttl = (f"Densest: {RACE_X[z.columns[ir]]}, age {AGE_LAB[z.index[ia]]} ({z.values.max():.0f}%) "
               f"· falls with age")
    return title(fig, ttl)


# 3c
@app.callback(Output("g_bmi", "figure"), Input("years", "value"))
def g_bmi(yrs):
    df = sub(yrs)
    df = df[(df.bmi != 9) & (df.dens != 9)]
    s = std_by(df, "bmi", "dn", "dd")
    labs = ["BMI < 25", "BMI 25–29.9", "BMI 30–34.9", "BMI 35+"]
    fig = go.Figure(go.Bar(x=labs, y=s.values, marker=dict(color=BMI_RAMP, line=dict(color="white", width=2)),
                           text=[f"{v:.0f}%" for v in s.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br><b>%{y:.1f}%</b> dense (age-standardised)<extra></extra>"))
    base(fig, 340, legend=False, t=60)
    fig.update_yaxes(title_text="% dense (age-standardised)", range=[0, max(s.values) * 1.2])
    return title(fig, f"Dense share: {s.iloc[0]:.0f}% at BMI < 25 → {s.iloc[-1]:.0f}% at BMI 35+")


# 4a
FACTORS = [
    ("Family history", "fh", {0: "No", 1: "Yes"}),
    ("Breast density", "dens", {1: "A fat", 2: "B scattered", 3: "C hetero.", 4: "D extreme"}),
    ("BMI", "bmi", {1: "< 25", 2: "25–29.9", 3: "30–34.9", 4: "35+"}),
    ("Hormone therapy †", "hrt", {0: "Not current", 1: "Current"}),
    ("Menopause †", "meno", {1: "Pre / peri", 2: "Post", 3: "Surgical"}),
    ("Race / ethnicity", "race", {c: v[0] for c, v in RACE.items()}),
]


@app.callback(Output("g_lolli", "figure"), Input("years", "value"))
def g_lolli(yrs):
    df = sub(yrs)
    df = df[df.bch != 9]
    overall = pct(df.sv.sum(), df.svd.sum())
    fac, lev, val, nn = [], [], [], []
    for fname, col, levels in FACTORS:
        d = df[df[col] != 9]
        s = std_by(d, col, "sv", "svd")
        cnt = d.groupby(col).n.sum()
        for code, lab in levels.items():
            if code in s.index:
                fac.append(fname)
                lev.append(lab)
                val.append(s[code])
                nn.append(cnt[code])
    fig = go.Figure()
    xs, ys0, ys1 = [], [], []
    for v, f, l in zip(val, fac, lev):
        xs += [0, v, None]
        ys0 += [f, f, None]
        ys1 += [l, l, None]
    fig.add_trace(go.Scatter(x=xs, y=[ys0, ys1], mode="lines", line=dict(color=GRID2, width=3),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=val, y=[fac, lev], mode="markers+text", showlegend=False,
                             marker=dict(color=ROSE, size=12, line=dict(color="white", width=2)),
                             text=[f"  {v:.1f}%" for v in val], textposition="middle right",
                             customdata=np.array(nn), textfont=dict(size=12, color=INK),
                             hovertemplate="%{y[0]} · %{y[1]}<br>Prior breast cancer: <b>%{x:.1f}%</b>"
                                           "<br>%{customdata:,.0f} exams<extra></extra>"))
    fig.add_vline(x=overall, line=dict(color=PLUM, width=1.5, dash="dash"))
    fig.add_annotation(x=overall, y=1.0, yref="paper", text=f"overall {overall:.1f}%", showarrow=False,
                       yanchor="bottom", font=dict(size=12, color=PLUM))
    base(fig, 520, l=150, r=60, t=64, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="% of exams from women with a prior breast cancer (age-standardised)",
                     range=[0, max(val) * 1.18], showgrid=True, gridcolor=GRID)
    fam = dict(zip(zip(fac, lev), val))
    fy, fn = fam[("Family history", "Yes")], fam[("Family history", "No")]
    return title(fig, f"Family history: {fy:.1f}% vs {fn:.1f}% — about {fy / fn:.1f}× higher at the same age")


# 4b
@app.callback(Output("g_famage", "figure"), Input("years", "value"))
def g_famage(yrs):
    df = sub(yrs)
    df = df[(df.bch != 9) & (df.fh != 9) & (df.age >= 4)]
    g = df.groupby(["age", "fh"])[["sv", "svd"]].sum()
    r = (100 * g.sv / g.svd).unstack()
    nmat = df.groupby(["age", "fh"]).n.sum().unstack()
    fig = go.Figure()
    for code, nm, col in ((0, "No family history", BLUE), (1, "Family history", ROSE)):
        fig.add_trace(go.Scatter(x=[AGE_LAB[a] for a in r.index], y=r[code], mode="lines+markers", name=nm,
                                 line=dict(color=col, width=2.5), customdata=nmat[code],
                                 marker=dict(size=8, line=dict(color="white", width=2)),
                                 hovertemplate=nm + "<br>Age %{x}: <b>%{y:.1f}%</b><br>%{customdata:,.0f} exams"
                                 "<extra></extra>"))
        fig.add_annotation(x=AGE_LAB[r.index[-1]], y=r[code].iloc[-1], text=f" {nm}", showarrow=False,
                           xanchor="left", font=dict(size=12, color=INK))
    base(fig, 380, r=130, legend=False)
    fig.update_yaxes(title_text="% with a prior breast cancer", rangemode="tozero")
    fig.update_xaxes(title_text="Age group (years)")
    gap = r[1] - r[0]
    ratio = (r[1] / r[0]).loc[5:].mean()
    return title(fig, f"Gap widens with age: +{gap.loc[5]:.1f} pts at 45–49 → +{gap.loc[13]:.1f} pts at 85+ "
                      f"(≈{ratio:.1f}× throughout)")


# 5a
@app.callback(Output("g_surv", "figure"), Input("years", "value"))
def g_surv(yrs):
    if yrs[0] == yrs[1]:
        return empty("Select two different years to compare.")
    df = sub(yrs)
    df = df[df.bch != 9]
    crude = rate_by(df, "year", "sv", "svd")
    adj = std_by(df, "year", "sv", "svd")
    fig = go.Figure()
    for nm, s, col, ysh in (("As observed", crude, ROSE, 9), ("Age mix held fixed", adj, BLUE, -9)):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=nm,
                                 line=dict(color=col, width=2.5),
                                 marker=dict(size=8, line=dict(color="white", width=2)),
                                 hovertemplate=nm + "<br>%{x}: <b>%{y:.2f}%</b><extra></extra>"))
        fig.add_annotation(x=s.index[-1], y=s.values[-1], text=f" {nm}", showarrow=False, xanchor="left",
                           yshift=ysh, font=dict(size=12, color=INK))
    base(fig, 380, r=130, legend=False)
    fig.update_yaxes(title_text="Survivors, % of exams", rangemode="tozero")
    fig.update_xaxes(dtick=2, range=[yrs[0] - 0.3, yrs[1] + 0.3])
    d_all = crude.iloc[-1] - crude.iloc[0]
    d_adj = adj.iloc[-1] - adj.iloc[0]
    share = 100 * d_adj / d_all if d_all else np.nan
    if d_all >= 0.5 and 0 <= share <= 100:
        return title(fig, f"Rise of {d_all:+.1f} pts; ageing explains only ~{100 - share:.0f}% of it")
    return title(fig, f"Survivors {crude.iloc[0]:.1f}% → {crude.iloc[-1]:.1f}% ({d_all:+.1f} pts observed, "
                      f"{d_adj:+.1f} with age mix fixed)")


# 5b
@app.callback(Output("g_survheat", "figure"), Input("years", "value"), Input("surv_mode", "value"))
def g_survheat(yrs, mode):
    df = sub(yrs)
    df = df[(df.bch != 9) & (df.age >= 4)]
    g = df.groupby(["age", "year"])[["sv", "svd"]].sum()
    if mode == "pct":
        z = (100 * g.sv / g.svd).unstack()
        fig = heat(z.values, [str(y) for y in z.columns], [AGE_LAB[a] for a in z.index],
                   [[0, "#FBEAF0"], [0.5, "#E07C9B"], [1, PLUM]], fmt="{:.0f}", cut=0.45,
                   hover="Age %{y} · %{x}<br><b>%{z:.1f}%</b> of exams", cbar="% survivors",
                   xtitle="", ytitle="Age")
        ttl = f"Highest cell: {np.nanmax(z.values):.0f}% of exams are from survivors"
    else:
        z = g.sv.unstack()
        fig = heat(z.values, [str(y) for y in z.columns], [AGE_LAB[a] for a in z.index],
                   [[0, "#FBEAF0"], [0.5, "#E07C9B"], [1, PLUM]],
                   fmt=lambda v: f"{v / 1000:.1f}k" if v >= 1000 else f"{v:.0f}", cut=0.45,
                   hover="Age %{y} · %{x}<br><b>%{z:,.0f}</b> survivor exams", cbar="exams", xtitle="", ytitle="Age")
        ttl = "Survivor exams peak in the 55–74 screening ages"
    return title(fig, ttl)


# 5c
@app.callback(Output("g_profile", "figure"), Input("years", "value"), Input("prof_mode", "value"))
def g_profile(yrs, mode):
    df = sub(yrs)
    df = df[df.bch != 9]
    rows = []
    for lab, (num, den) in TRAITS.items():
        if lab in ("Prior breast cancer", "Prior biopsy") or (mode == "adj" and lab == "Age 65+"):
            continue
        if mode == "adj":
            d = df[df[den] > 0]
            s = std_by(d, "bch", num, den)
            rows.append((lab, s.get(0, np.nan), s.get(1, np.nan)))
        else:
            r0 = pct(df[df.bch == 0][num].sum(), df[df.bch == 0][den].sum())
            r1 = pct(df[df.bch == 1][num].sum(), df[df.bch == 1][den].sum())
            rows.append((lab, r0, r1))
    rows.sort(key=lambda r: -(r[2] - r[1]))
    lab, a, b = zip(*rows)
    fig = dumbbell(list(lab), list(a), list(b), "No prior cancer", "Survivors", BLUE, ROSE, h=380, l=190)
    big = rows[0]
    small = rows[-1]
    ttl = (f"Survivors differ most on {big[0].lower()} ({big[2] - big[1]:+.1f} pts)" if big[2] - big[1] >= abs(small[2] - small[1])
           else f"Survivors differ most on {small[0].lower()} ({small[2] - small[1]:+.1f} pts)")
    return title(fig, ttl)


# 6a
@app.callback(Output("g_tree", "figure"), Input("years", "value"), Input("tree_metric", "value"))
def g_tree(yrs, metric):
    num, den = TRAITS[metric]
    df = sub(yrs)
    df = df[df.dens != 9].assign(dg=lambda x: np.where(x.dens >= 3, "Dense (C/D)", "Not dense (A/B)"))
    g = df.groupby(["band", "dg"], observed=True)[["n", num, den]].sum().reset_index()
    ids, labs, par, vals, cols = ["all"], ["All exams"], [""], [g.n.sum()], [pct(g[num].sum(), g[den].sum())]
    for bnd in BANDS:
        gb = g[g.band == bnd]
        if gb.empty:
            continue
        ids.append(bnd); labs.append(bnd); par.append("all"); vals.append(gb.n.sum())
        cols.append(pct(gb[num].sum(), gb[den].sum()))
        for _, r in gb.iterrows():
            ids.append(f"{bnd}|{r.dg}"); labs.append(r.dg); par.append(bnd); vals.append(r.n)
            cols.append(pct(r[num], r[den]))
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labs, parents=par, values=vals, branchvalues="total",
        marker=dict(colors=cols, colorscale=[[0, "#FBEAF0"], [0.5, "#E07C9B"], [1, PLUM]],
                    cmin=np.nanmin(cols), cmax=np.nanmax(cols), line=dict(width=2, color="white"),
                    colorbar=dict(title=dict(text="%", side="right"), thickness=12, len=0.85)),
        customdata=np.round(cols, 1), texttemplate="<b>%{label}</b><br>%{customdata}%",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} exams<br>" + metric + ": <b>%{customdata}%</b>"
                      "<extra></extra>", textfont=dict(family=FONT, size=13), pathbar=dict(visible=True)))
    base(fig, 440, l=10, r=10, t=44, b=10, legend=False)
    top = max(range(1, len(ids)), key=lambda i: vals[i] if par[i] != "all" and par[i] != "" else -1)
    return title(fig, f"Biggest segment: {labs[top]} in {par[top]} ({100 * vals[top] / vals[0]:.0f}% of exams)")


# 6b
@app.callback(Output("g_bubble", "figure"), Input("years", "value"))
def g_bubble(yrs):
    df = sub(yrs)
    df = df[df.race != 9]
    g = df.groupby(["band", "race"], observed=True)[["n", "dn", "dd", "sv", "svd"]].sum().reset_index()
    g = g[g.n >= 5000]
    g["x"] = 100 * g.dn / g.dd
    g["y"] = 100 * g.sv / g.svd
    fig = go.Figure()
    sizeref = 2.0 * g.n.max() / (46 ** 2)
    for i, bnd in enumerate(BANDS):
        s = g[g.band == bnd]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.x, y=s.y, mode="markers+text", name=bnd, text=[RACE_SHORT[r] for r in s.race],
            textfont=dict(size=11, color="white"), textposition="middle center",
            marker=dict(size=s.n, sizemode="area", sizeref=sizeref, sizemin=11, color=AGE_RAMP[i],
                        line=dict(color="white", width=2)),
            customdata=np.c_[[RACE[r][0] for r in s.race], s.n],
            hovertemplate="<b>%{customdata[0]}, " + bnd + "</b><br>Dense: %{x:.1f}%<br>Prior cancer: %{y:.1f}%"
                          "<br>%{customdata[1]:,.0f} exams<extra></extra>"))
    base(fig, 440)
    fig.update_xaxes(title_text="% with dense breasts", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text="% with a prior breast cancer")
    fig.add_annotation(text="W White · B Black · A Asian/PI · H Hispanic · O Other", xref="paper", yref="paper",
                       x=1, y=1.02, showarrow=False, xanchor="right", yanchor="bottom",
                       font=dict(size=11, color=MUTED))
    return title(fig, "Older segments hold more survivors; Asian / PI segments are densest")


# 6c
P_RACE = {v[0]: c for c, v in RACE.items()}


@app.callback(Output("persona_tiles", "children"), Output("g_persona", "figure"), Input("years", "value"),
              Input("p_age", "value"), Input("p_race", "value"), Input("p_dens", "value"),
              Input("p_bmi", "value"), Input("p_fh", "value"))
def persona(yrs, age, race, dens, bmi, fh):
    df = sub(yrs)
    m = np.ones(len(df), dtype=bool)
    if age != "All":
        m &= (df.band == age).values
    if race != "All":
        m &= (df.race == P_RACE[race]).values
    if dens == "Dense (C/D)":
        m &= df.dens.isin([3, 4]).values
    elif dens == "Not dense (A/B)":
        m &= df.dens.isin([1, 2]).values
    if bmi == "BMI < 30":
        m &= df.bmi.isin([1, 2]).values
    elif bmi == "BMI ≥ 30":
        m &= df.bmi.isin([3, 4]).values
    if fh != "All":
        m &= (df.fh == (1 if fh == "Yes" else 0)).values
    seg = df[m]
    if seg.n.sum() < 1:
        return [html.P("No exams match this profile — loosen a filter.", className="card-note")], \
            empty("No exams match this profile.")
    share = 100 * seg.n.sum() / df.n.sum()

    def cmp(label, num, den, good_up=None):
        v, ref = pct(seg[num].sum(), seg[den].sum()), pct(df[num].sum(), df[den].sum())
        return tile(label, "–" if np.isnan(v) else f"{v:.1f}%", f"all women {ref:.1f}%")

    tiles = [tile("Exams in segment", f"{seg.n.sum():,.0f}", f"{share:.1f}% of all exams in view", "rose"),
             cmp("Prior breast cancer", "sv", "svd"), cmp("Dense breasts", "dn", "dd"),
             cmp("Obese (BMI ≥ 30)", "ob", "obd"), cmp("Hormone therapy", "hr", "hrd")]
    s = 100 * seg.groupby("year").n.sum() / df.groupby("year").n.sum()
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines+markers", line=dict(color=ROSE, width=2.5),
                               fill="tozeroy", fillcolor="rgba(194,69,110,0.12)",
                               marker=dict(size=7, line=dict(color="white", width=2)),
                               hovertemplate="%{x}: <b>%{y:.2f}%</b> of all exams<extra></extra>"))
    base(fig, 260, t=48, legend=False)
    fig.update_yaxes(title_text="% of all exams", rangemode="tozero")
    fig.update_xaxes(dtick=2, range=[yrs[0] - 0.3, yrs[1] + 0.3])
    trend = s.iloc[-1] - s.iloc[0]
    return tiles, title(fig, f"Segment share of exams: {s.iloc[0]:.1f}% → {s.iloc[-1]:.1f}% ({trend:+.1f} pts)")


# 7
@app.callback(Output("takeaways", "children"), Input("years", "value"))
def takeaways(yrs):
    df = sub(yrs)
    k = df[df.bch != 9]
    c = rate_by(k, "year", "sv", "svd")
    a = std_by(k, "year", "sv", "svd")
    dn = pct(df.dn.sum(), df.dd.sum())
    d0, d1 = df[df.year == yrs[0]], df[df.year == yrs[1]]
    nw0, nw1 = pct(d0.nw.sum(), d0.rd.sum()), pct(d1.nw.sum(), d1.rd.sum())
    ob0, ob1 = pct(d0.ob.sum(), d0.obd.sum()), pct(d1.ob.sum(), d1.obd.sum())
    hr0, hr1 = pct(d0.hr.sum(), d0.hrd.sum()), pct(d1.hr.sum(), d1.hrd.sum())
    cards = [
        ("Survivors", f"{c.iloc[0]:.1f}% → {c.iloc[-1]:.1f}%",
         f"Survivors’ share of exams {yrs[0]}→{yrs[1]}. Holding age fixed it is still {a.iloc[0]:.1f}% → "
         f"{a.iloc[-1]:.1f}%, so this is more than an ageing effect.",
         "Build a survivorship pathway: recall, follow-up imaging, support."),
        ("Density", f"{dn:.0f}% dense",
         "Nearly half of exams are of dense breasts — most common in younger women and in Asian / PI women at "
         "every age.", "Prepare for demand for supplemental imaging and for ‘what does dense mean?’ conversations."),
        ("The changing patient", f"BMI≥30 {ob0:.0f}% → {ob1:.0f}%",
         f"HRT use {hr0:.1f}% → {hr1:.1f}%, non-White share {nw0:.0f}% → {nw1:.0f}%.",
         "Update personas, multilingual outreach, comfort and equipment planning."),
        ("Targeting", "Segment first",
         "Age × density × ethnicity segments differ sharply in size and survivor share (chapter 7).",
         "Size a segment with the persona builder before you campaign."),
    ]
    return [html.Div([html.Div(h, className="tk-h"), html.Div(big, className="tk-big"), html.P(txt),
                      html.Div(["Action: ", act], className="tk-act")], className="take") for h, big, txt, act in cards]

# ───────────────────────── geography (chapter 2) ─────────────────────────
SMALL = {"DC", "RI", "DE", "CT", "NJ", "NH", "VT", "MA", "MD"}


@app.callback(Output("g_map", "figure"), Input("geo_metric", "value"), Input("geo_state", "value"))
def g_map(metric, picked):
    lab, short, f = GEO[metric]
    v = STATE[metric].values
    lo, hi = float(v.min()), float(v.max())
    rank = STATE[metric].rank(ascending=False, method="min").astype(int).values
    fig = go.Figure()
    txt_col, lx, ly, lt = [], [], [], []
    for i, row in STATE.iterrows():
        rings = SHAPES[row.state]
        xs, ys = [], []
        for x, y in rings:
            xs += list(x) + [x[0], None]
            ys += list(y) + [y[0], None]
        t = (row[metric] - lo) / (hi - lo)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself", fillcolor=ramp(t), line=dict(color="white", width=0.8),
            hoveron="fills", hoverinfo="text", name=row.state, showlegend=False,
            text=f"<b>{row.state}</b><br>{short}: <b>{f.format(row[metric])}</b><br>rank {rank[i]} of 51 (1 = highest)"))
        if row.abbr not in SMALL:
            cx, cy = LABELS[row.state]
            lx.append(cx); ly.append(cy); lt.append(row.abbr); txt_col.append("white" if t > 0.45 else INK)
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="text", text=lt, hoverinfo="skip", showlegend=False,
                             textfont=dict(size=10, color=txt_col)))
    if picked in SHAPES:
        xs, ys = [], []
        for x, y in SHAPES[picked]:
            xs += list(x) + [x[0], None]
            ys += list(y) + [y[0], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=INK, width=3),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(colorscale=[[k / (len(MAP_RAMP) - 1), c] for k, c in enumerate(MAP_RAMP)],
                                         cmin=lo, cmax=hi, color=[lo], showscale=True,
                                         colorbar=dict(thickness=12, len=0.55, x=0.86, y=0.4,
                                                       title=dict(text=short, side="right"),
                                                       tickformat=",.0f" if metric == "income" else ".0f"))))
    base(fig, 640, l=10, r=10, t=44, b=10, legend=False)
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(visible=False, showgrid=False, scaleanchor="x", scaleratio=1)
    top, bot = STATE.loc[v.argmax()], STATE.loc[v.argmin()]
    return title(fig, f"{lab}: highest in {top.state} ({f.format(top[metric])}), lowest in {bot.state} ({f.format(bot[metric])})")


@app.callback(Output("geo_state", "value"), Input("g_map", "clickData"), Input("g_geo_sc", "clickData"),
              prevent_initial_call=True)
def geo_pick(m, sc):
    if ctx.triggered_id == "g_map" and m:
        i = m["points"][0].get("curveNumber", -1)
        if 0 <= i < len(STATE):
            return STATE.state.iloc[i]
    if ctx.triggered_id == "g_geo_sc" and sc:
        return sc["points"][0]["customdata"]
    raise PreventUpdate


@app.callback(Output("geo_tiles", "children"), Input("geo_state", "value"))
def geo_tiles(st):
    r = STATE[STATE.state == st].iloc[0]
    out = []
    for k, cls in (("inc", ""), ("mort", ""), ("mi", "rose"), ("obese", ""), ("unins", "")):
        lab, short, f = GEO[k]
        rk = int((STATE[k] > r[k]).sum()) + 1
        out.append(tile(lab.split(" (")[0], f.format(r[k]), f"rank {rk} of 51 · state average {f.format(STATE[k].mean())}", cls))
    return out


@app.callback(Output("g_geo_sc", "figure"), Input("geo_x", "value"), Input("geo_y", "value"), Input("geo_state", "value"))
def g_geo_sc(xk, yk, picked):
    xl, xs_, xf = GEO[xk]
    yl, ys_, yf = GEO[yk]
    x, y = STATE[xk].values, STATE[yk].values
    r = float(np.corrcoef(x, y)[0, 1])
    m, b0 = np.polyfit(x, y, 1)
    fig = go.Figure()
    xx = np.array([x.min(), x.max()])
    fig.add_trace(go.Scatter(x=xx, y=m * xx + b0, mode="lines", line=dict(color=GRID2, width=2, dash="dash"),
                             hoverinfo="skip", showlegend=False))
    sel = (STATE.state == picked).values
    fig.add_trace(go.Scatter(
        x=x[~sel], y=y[~sel], mode="markers", customdata=STATE.state[~sel], showlegend=False,
        marker=dict(color=ROSE, size=10, opacity=0.85, line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{customdata}</b><br>" + xs_ + ": %{x}<br>" + ys_ + ": %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x[sel], y=y[sel], mode="markers+text", customdata=STATE.state[sel], showlegend=False, text=STATE.state[sel],
        textposition="top center", textfont=dict(size=12, color=INK),
        marker=dict(color=PLUM, size=14, line=dict(color="white", width=2)),
        hovertemplate="<b>%{customdata}</b><br>" + xs_ + ": %{x}<br>" + ys_ + ": %{y}<extra></extra>"))
    extra = {int(np.argmax(y)), int(np.argmin(y)), int(np.argmax(x)), int(np.argmin(x))}
    extra = [i for i in extra if STATE.state[i] != picked]
    fig.add_trace(go.Scatter(x=x[extra], y=y[extra], mode="text", text=STATE.abbr[extra], textposition="top right",
                             textfont=dict(size=11, color=MUTED), hoverinfo="skip", showlegend=False))
    base(fig, 440, l=70, r=30, legend=False)
    fig.update_xaxes(title_text=xl, showgrid=True, gridcolor=GRID, tickformat=",.0f" if xk == "income" else None)
    fig.update_yaxes(title_text=yl)
    if abs(r) < 0.25:
        head = f"Across states, {xs_} and {ys_} show little link (r = {r:+.2f})"
    else:
        head = (f"States with higher {xs_} tend to have {'higher' if r > 0 else 'lower'} {ys_} "
                f"(r = {r:+.2f}{', strong' if abs(r) >= 0.6 else ''})")
    return title(fig, head)



if __name__ == "__main__":
    # debug + hot reload: save app.py and the browser refreshes by itself (no error bubble on screen)
    app.run(debug=True, dev_tools_ui=False, dev_tools_props_check=False)
