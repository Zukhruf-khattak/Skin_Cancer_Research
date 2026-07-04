"""
🏥  SkinSense AI — SK1 × Final Stage 2 Demo
BSc Final Year Project | University of Peshawar
Zohrouf Khattak & Huma Zeb | Supervisor: Dr Muhammad Ayaz

Paths used
  Stage 1 weights  →  SK1/
  Stage 2 weights  →  Final Stage 2/
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, time
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinSense AI · SK1 × Stage 2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
SK1  = os.path.join(BASE, "SK1")            # Stage 1 weights live here
FS2  = os.path.join(BASE, "Final Stage 2")  # Stage 2 weights live here

MODEL_PATHS_S1 = {
    "ResNet50":        os.path.join(SK1, "best_resnet50.pth"),
    "EfficientNet-B0": os.path.join(SK1, "best_efficientnet-b0.pth"),
    "MobileNetV2":     os.path.join(SK1, "best_mobilenetv2.pth"),
    "ConvNeXt-Tiny":   os.path.join(SK1, "best_convnext-tiny.pth"),
    "Swin-Tiny":       os.path.join(SK1, "best_swin-tiny.pth"),
}
MODEL_PATHS_S2 = {
    "ResNet50":        os.path.join(FS2, "ResNet50_stage2.pth"),
    "EfficientNet-B0": os.path.join(FS2, "EfficientNet-B0_stage2.pth"),
    "MobileNetV2":     os.path.join(FS2, "MobileNetV2_stage2.pth"),
    "ConvNeXt-Tiny":   os.path.join(FS2, "ConvNeXt-Tiny_stage2.pth"),
    "Swin-Tiny":       os.path.join(FS2, "Swin-Tiny_stage2.pth"),
}

# ─────────────────────────────────────────────────────────────
# PYTORCH
# ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    import torchvision.transforms as T
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

:root {
  --bg:       #07090f;
  --card:     #0f1623;
  --card2:    #151f2e;
  --accent:   #6366f1;
  --acc2:     #8b5cf6;
  --cyan:     #06b6d4;
  --green:    #10b981;
  --amber:    #f59e0b;
  --rose:     #f43f5e;
  --text:     #f1f5f9;
  --muted:    #94a3b8;
  --border:   rgba(99,102,241,0.18);
  --border2:  rgba(99,102,241,0.08);
}

html, body, [class*="css"] { font-family:'Inter',sans-serif; color:var(--text); }
.stApp { background:var(--bg); }
.block-container { padding-top:0 !important; max-width:1320px; }
#MainMenu, footer, header { visibility:hidden; }
[data-testid="collapsedControl"], [data-testid="stSidebar"] { display:none !important; }

/* ── TOP NAV ── */
.topnav {
  display:flex; align-items:center; gap:0.3rem;
  background:rgba(15,22,35,0.97);
  border-bottom:1px solid var(--border);
  padding:0.65rem 1.8rem; position:sticky; top:0; z-index:9999;
  backdrop-filter:blur(16px); flex-wrap:wrap;
}
.topnav-brand {
  font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:800;
  background:linear-gradient(135deg,#a5b4fc,#38bdf8);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin-right:1.5rem; white-space:nowrap; letter-spacing:-0.02em;
}
.topnav a {
  color:var(--muted); text-decoration:none; font-size:0.84rem; font-weight:500;
  padding:0.38rem 0.9rem; border-radius:8px;
  transition:background 0.15s,color 0.15s; white-space:nowrap;
}
.topnav a:hover  { background:rgba(99,102,241,0.14); color:#c7d2fe; }
.topnav a.active { background:linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff !important; font-weight:600; }
.topnav-badge {
  margin-left:auto; font-size:0.72rem; font-weight:600; letter-spacing:.04em;
  background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.35);
  color:#a5b4fc; border-radius:100px; padding:.2rem .75rem; white-space:nowrap;
}

/* ── HERO ── */
.hero {
  background:linear-gradient(135deg,#0e0c2f,#1a1060 28%,#0a1628 60%,#07090f);
  border:1px solid var(--border); border-radius:20px;
  padding:2.6rem 3rem 2.2rem; margin:1.1rem 0 1.6rem;
  position:relative; overflow:hidden;
}
.hero::before {
  content:''; position:absolute; inset:0;
  background:radial-gradient(ellipse at 20% 60%,rgba(99,102,241,.13),transparent 55%),
              radial-gradient(ellipse at 78% 40%,rgba(56,189,248,.08),transparent 55%);
  animation:heroGlow 7s ease-in-out infinite alternate;
}
@keyframes heroGlow { 0%{opacity:.5} 100%{opacity:1} }
.hero-title {
  font-family:'Space Grotesk',sans-serif; font-size:2.4rem; font-weight:800;
  background:linear-gradient(135deg,#c7d2fe,#818cf8 40%,#38bdf8 80%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  position:relative; z-index:1; margin:0 0 .4rem; letter-spacing:-0.03em;
}
.hero-sub  { font-size:.95rem; color:#94a3b8; position:relative; z-index:1; line-height:1.7; }
.hero-tags { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem; position:relative; z-index:1; }
.tag {
  background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.3);
  border-radius:100px; padding:.22rem .85rem; font-size:.74rem; color:#a5b4fc; font-weight:500;
}
.tag-green { background:rgba(16,185,129,.12); border-color:rgba(16,185,129,.3); color:#6ee7b7; }
.tag-amber { background:rgba(245,158,11,.12); border-color:rgba(245,158,11,.3); color:#fcd34d; }
.tag-cyan  { background:rgba(6,182,212,.12);  border-color:rgba(6,182,212,.3);  color:#67e8f9; }

/* ── KPI CARDS ── */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin:1rem 0; }
.kpi {
  background:var(--card); border:1px solid var(--border); border-radius:16px;
  padding:1.4rem 1rem; text-align:center;
  transition:transform .2s,box-shadow .2s;
}
.kpi:hover { transform:translateY(-4px); box-shadow:0 12px 36px rgba(99,102,241,.18); }
.kpi-icon { font-size:1.6rem; margin-bottom:.4rem; }
.kpi-val  {
  font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:800;
  background:linear-gradient(135deg,#a5b4fc,#38bdf8);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.kpi-val-green { background:linear-gradient(135deg,#6ee7b7,#10b981); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.kpi-val-amber { background:linear-gradient(135deg,#fcd34d,#f59e0b); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.kpi-val-rose  { background:linear-gradient(135deg,#fda4af,#f43f5e); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.kpi-lbl  { font-size:.72rem; color:var(--muted); margin-top:.28rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
.kpi-delta { font-size:.73rem; color:var(--green); margin-top:.3rem; }

/* ── SECTION TITLE ── */
.sec {
  font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:700;
  color:#e2e8f0; padding-bottom:.4rem; border-bottom:1px solid var(--border);
  margin:1.8rem 0 .9rem; letter-spacing:-0.01em;
}
.sec-sm { font-size:.95rem; font-weight:600; color:#a5b4fc; margin:.9rem 0 .4rem; }

/* ── CARDS ── */
.card {
  background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.3rem;
  transition:border-color .2s;
}
.card:hover { border-color:rgba(99,102,241,.4); }
.card h4 { font-family:'Space Grotesk',sans-serif; font-size:.92rem; font-weight:700; color:#a5b4fc; margin:0 0 .4rem; }
.card p  { font-size:.83rem; color:#94a3b8; line-height:1.7; margin:0; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background:var(--card); border-radius:11px; padding:.25rem;
  border:1px solid var(--border); gap:.2rem;
}
.stTabs [data-baseweb="tab"] { border-radius:8px; color:var(--muted) !important; font-weight:500; padding:.45rem 1.1rem; font-size:.88rem; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,var(--accent),var(--acc2)) !important; color:#fff !important; font-weight:600 !important; }

/* ── PREDICTION CARD ── */
.pred-card {
  background:linear-gradient(135deg,#0d0c2b,#111e35);
  border:1px solid rgba(99,102,241,.45); border-radius:18px; padding:2rem 1.8rem; text-align:center;
}
.pred-label { font-size:1.55rem; font-weight:800; margin-bottom:.35rem; font-family:'Space Grotesk',sans-serif; }
.conf-bar { background:rgba(255,255,255,.08); border-radius:100px; height:10px; overflow:hidden; margin:.8rem 0; }
.conf-fill { height:100%; border-radius:100px; transition:width 1s cubic-bezier(.4,0,.2,1); }

/* ── ALERTS ── */
.warn { background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.3); border-radius:11px; padding:.9rem 1.1rem; font-size:.83rem; color:#fcd34d; margin-bottom:1.2rem; line-height:1.6; }
.err  { background:rgba(244,63,94,.08);  border:1px solid rgba(244,63,94,.3);  border-radius:11px; padding:.9rem 1.1rem; font-size:.83rem; color:#fda4af; margin-bottom:1rem; }
.info { background:rgba(6,182,212,.08);  border:1px solid rgba(6,182,212,.25); border-radius:11px; padding:.9rem 1.1rem; font-size:.83rem; color:#67e8f9; margin-bottom:1rem; }

/* ── MODEL PROB BARS ── */
.mbar { display:flex; align-items:center; gap:.7rem; margin:.32rem 0; }
.mbar-name  { width:148px; font-size:.8rem; color:#94a3b8; flex-shrink:0; }
.mbar-track { flex:1; background:#0f1623; border-radius:100px; height:7px; overflow:hidden; border:1px solid rgba(99,102,241,.1); }
.mbar-fill  { height:100%; border-radius:100px; transition:width .7s ease; }
.mbar-val   { width:46px; font-size:.8rem; color:#a5b4fc; text-align:right; flex-shrink:0; }

/* ── UPLOAD ── */
.up-hint {
  background:rgba(99,102,241,.04); border:2px dashed rgba(99,102,241,.25);
  border-radius:16px; padding:2rem 1.5rem; text-align:center;
  color:var(--muted); font-size:.86rem; line-height:1.8;
}
div[data-testid="stFileUploader"] {
  background:rgba(99,102,241,.03); border:2px dashed rgba(99,102,241,.22);
  border-radius:14px; padding:.3rem;
}

/* ── BUTTONS ── */
.stButton > button {
  background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
  border:none !important; border-radius:10px !important; color:#fff !important;
  font-weight:700 !important; padding:.6rem 1.8rem !important;
  transition:all .18s !important; font-family:'Inter',sans-serif !important;
}
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 6px 22px rgba(99,102,241,.45) !important; }
.stButton > button:disabled { opacity:.4 !important; transform:none !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }

/* ── PIPELINE STEP ── */
.pipeline {
  display:flex; align-items:stretch; gap:0; flex-wrap:wrap; margin:1rem 0;
}
.pipe-step {
  flex:1; min-width:110px; background:var(--card2);
  border:1px solid var(--border); padding:1rem .8rem; text-align:center;
  position:relative; font-size:.78rem; color:#94a3b8; line-height:1.5;
}
.pipe-step:not(:last-child)::after {
  content:'→'; position:absolute; right:-14px; top:50%; transform:translateY(-50%);
  color:#6366f1; font-size:1.1rem; z-index:1;
}
.pipe-step:first-child { border-radius:12px 0 0 12px; }
.pipe-step:last-child  { border-radius:0 12px 12px 0; }
.pipe-icon { font-size:1.4rem; margin-bottom:.35rem; }
.pipe-title { font-weight:700; color:#e2e8f0; font-size:.82rem; margin-bottom:.2rem; }

/* ── FOOTER ── */
.foot {
  text-align:center; color:#475569; font-size:.76rem; margin-top:2.5rem;
  padding-top:1.2rem; border-top:1px solid var(--border2); line-height:1.7;
}

/* ── BADGE ── */
.badge {
  display:inline-block; font-size:.67rem; font-weight:700; letter-spacing:.05em;
  text-transform:uppercase; padding:.18rem .55rem; border-radius:4px; margin-left:.4rem;
}
.badge-new   { background:rgba(16,185,129,.2);  color:#6ee7b7; }
.badge-best  { background:rgba(99,102,241,.25); color:#a5b4fc; }
.badge-focal { background:rgba(245,158,11,.2);  color:#fcd34d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE + NAVIGATION
# ─────────────────────────────────────────────────────────────
PAGES      = ["Overview", "Stage 1 (SK1)", "Final Stage 2", "Live Demo", "Gallery", "About"]
PAGE_ICONS = {"Overview":"🏠","Stage 1 (SK1)":"🔵","Final Stage 2":"🟣","Live Demo":"🔮","Gallery":"🖼️","About":"ℹ️"}

if "page" not in st.session_state:
    st.session_state.page = "Overview"

qp = st.query_params
if "page" in qp and qp["page"] in PAGES:
    st.session_state.page = qp["page"]

page = st.session_state.page

# Build nav bar
nav_html = '<div class="topnav"><div class="topnav-brand">🔬 SkinSense AI</div>'
for p in PAGES:
    cls = "active" if page == p else ""
    nav_html += f'<a class="{cls}" href="?page={p}" target="_self">{PAGE_ICONS[p]} {p}</a>'
nav_html += '<div class="topnav-badge">SK1 × Final Stage 2</div></div>'
st.markdown(nav_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HARD-CODED DATA — sourced from SK1/ & Final Stage 2/ CSVs
# ─────────────────────────────────────────────────────────────
# Stage 1 (SK1 folder) — from model_metrics_summary.csv
S1 = {
    "ResNet50":        {"Accuracy":92.76,"Precision":90.80,"Sensitivity":89.44,"F1":0.9011,"AUC":0.9675,"TP":1317,"TN":2222,"FP":177,"FN":84},
    "EfficientNet-B0": {"Accuracy":92.87,"Precision":88.38,"Sensitivity":92.86,"F1":0.9057,"AUC":0.9713,"TP":1315,"TN":2214,"FP":185,"FN":86},
    "MobileNetV2":     {"Accuracy":91.89,"Precision":87.61,"Sensitivity":90.86,"F1":0.8921,"AUC":0.9680,"TP":1284,"TN":2213,"FP":186,"FN":117},
    "ConvNeXt-Tiny":   {"Accuracy":93.18,"Precision":87.71,"Sensitivity":94.79,"F1":0.9111,"AUC":0.9746,"TP":1314,"TN":2229,"FP":170,"FN":87},
    "Swin-Tiny":       {"Accuracy":93.55,"Precision":88.79,"Sensitivity":94.43,"F1":0.9153,"AUC":0.9734,"TP":1316,"TN":2216,"FP":183,"FN":85},
    "⭐ Ensemble":     {"Accuracy":94.34,"Precision":91.12,"Sensitivity":93.79,"F1":0.9244,"AUC":0.9845,"TP":1314,"TN":2271,"FP":128,"FN":87},
}

# Stage 2 (Final Stage 2 folder) — from ultimate_summary_metrics.csv
S2 = {
    "ResNet50":        {"Accuracy":80.48,"MacroPrec":76.65,"MacroSens":81.88,"MacroF1":0.7814,"AUC":0.9481,"MEL":81.67,"BCC":77.33,"SCC":91.58,"AK":76.92,"CPS":0.8472},
    "EfficientNet-B0": {"Accuracy":83.39,"MacroPrec":79.99,"MacroSens":83.34,"MacroF1":0.8133,"AUC":0.9615,"MEL":86.00,"BCC":82.00,"SCC":88.42,"AK":76.92,"CPS":0.8566},
    "MobileNetV2":     {"Accuracy":80.24,"MacroPrec":76.18,"MacroSens":80.44,"MacroF1":0.7764,"AUC":0.9506,"MEL":82.33,"BCC":79.00,"SCC":87.37,"AK":73.08,"CPS":0.8309},
    "ConvNeXt-Tiny":   {"Accuracy":80.36,"MacroPrec":76.47,"MacroSens":81.90,"MacroF1":0.7789,"AUC":0.9549,"MEL":80.67,"BCC":77.67,"SCC":91.58,"AK":77.69,"CPS":0.8443},
    "Swin-Tiny":       {"Accuracy":82.91,"MacroPrec":79.03,"MacroSens":84.41,"MacroF1":0.8063,"AUC":0.9617,"MEL":83.00,"BCC":80.00,"SCC":91.58,"AK":83.08,"CPS":0.8614},
    "⭐ Ensemble":     {"Accuracy":85.58,"MacroPrec":81.68,"MacroSens":86.42,"MacroF1":0.8332,"AUC":0.9738,"MEL":85.00,"BCC":85.67,"SCC":95.79,"AK":79.23,"CPS":0.8881},
}

CLS_NAMES  = ["Melanoma (MEL)", "Basal Cell Carcinoma (BCC)", "Squamous Cell Carcinoma (SCC)", "Actinic Keratosis (AK)"]
CLS_COLORS = ["#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4"]
CLS_ABBR   = ["MEL", "BCC", "SCC", "AK"]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def load_img(path):
    try:
        if os.path.exists(path):
            return Image.open(path)
    except Exception:
        pass
    return None

def gauge(value, title, color="#6366f1"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 12, "color": "#94a3b8", "family": "Inter"}},
        number={"suffix": "%", "font": {"size": 24, "color": "#e2e8f0", "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#475569"}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#0f1623", "bordercolor": "#1e293b",
            "steps": [
                {"range": [0, 60],  "color": "rgba(244,63,94,.06)"},
                {"range": [60, 80], "color": "rgba(245,158,11,.06)"},
                {"range": [80, 100],"color": "rgba(16,185,129,.06)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.85, "value": value},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=200, margin=dict(t=35, b=0, l=12, r=12),
        font=dict(family="Inter"),
    )
    return fig

def bar_chart(data, metric, title, palette=None):
    names = list(data.keys())
    vals  = [data[n][metric] for n in names]
    if palette is None:
        palette = ["#6366f1","#8b5cf6","#818cf8","#06b6d4","#38bdf8","#f59e0b"]
    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker_color=palette,
        marker_line_color="rgba(0,0,0,0)",
        text=[f"{v:.2f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#111827", color="#94a3b8", tickangle=-15, showgrid=False),
        yaxis=dict(gridcolor="#1e2a35", color="#94a3b8"),
        height=330, margin=dict(t=22, b=60, l=44, r=20),
        showlegend=False,
        title=dict(text=title, font=dict(color="#e2e8f0", size=13, family="Space Grotesk"), x=0),
        font=dict(family="Inter"),
    )
    return fig

def radar_chart_s2():
    models = list(S2.keys())
    cats   = ["Accuracy","MacroF1×100","AUC×100","MEL","BCC","SCC","AK"]
    fig = go.Figure()
    clrs = ["#6366f1","#8b5cf6","#818cf8","#06b6d4","#38bdf8","#f59e0b"]
    for i, (m, c) in enumerate(zip(models, clrs)):
        d = S2[m]
        vals = [d["Accuracy"], d["MacroF1"]*100, d["AUC"]*100, d["MEL"], d["BCC"], d["SCC"], d["AK"]]
        fill_clr = c[:1] + c[1:].replace("#", "") 
        # Build rgba fill colour from hex
        r_int = int(c[1:3], 16); g_int = int(c[3:5], 16); b_int = int(c[5:7], 16)
        fill_rgba = f"rgba({r_int},{g_int},{b_int},0.06)"
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            name=m, line=dict(color=c, width=2),
            fill="toself", fillcolor=fill_rgba,
        ))
    fig.update_layout(
        polar=dict(bgcolor="#0f1623", radialaxis=dict(range=[60,100],color="#475569",gridcolor="#1e2a35"),
                   angularaxis=dict(color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)", height=380,
        margin=dict(t=30,b=20,l=60,r=60),
        legend=dict(bgcolor="rgba(15,22,35,.9)",bordercolor="#1e293b",borderwidth=1,font=dict(color="#e2e8f0",size=11)),
        font=dict(family="Inter"),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
def build_model(arch, nc, stage=1):
    if arch == "ResNet50":
        m = tvm.resnet50(weights=None)
        inf = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(inf, nc)) if stage == 1 else nn.Linear(inf, nc)
    elif arch == "EfficientNet-B0":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    elif arch == "MobileNetV2":
        m = tvm.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    elif arch == "ConvNeXt-Tiny":
        m = tvm.convnext_tiny(weights=None)
        inf = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Dropout(0.5), nn.Linear(inf, nc)) if stage == 1 else nn.Linear(inf, nc)
    elif arch == "Swin-Tiny":
        m = tvm.swin_t(weights=None)
        inf = m.head.in_features
        m.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(inf, nc)) if stage == 1 else nn.Linear(inf, nc)
    else:
        raise ValueError(arch)
    return m

@st.cache_resource(show_spinner=False)
def load_s1_models():
    if not TORCH_OK:
        return None
    out = {}
    for name, path in MODEL_PATHS_S1.items():
        if not os.path.exists(path):
            continue
        try:
            m = build_model(name, 2, stage=1)
            sd = torch.load(path, map_location="cpu", weights_only=True)
            sd = sd.get("model_state_dict", sd.get("state_dict", sd))
            m.load_state_dict(sd)
            m.eval()
            out[name] = m
        except Exception as e:
            st.session_state[f"load_err_s1_{name}"] = str(e)
    return out or None

@st.cache_resource(show_spinner=False)
def load_s2_models():
    if not TORCH_OK:
        return None
    out = {}
    for name, path in MODEL_PATHS_S2.items():
        if not os.path.exists(path):
            continue
        try:
            m = build_model(name, 4, stage=2)
            sd = torch.load(path, map_location="cpu", weights_only=True)
            sd = sd.get("model_state_dict", sd.get("state_dict", sd))
            m.load_state_dict(sd)
            m.eval()
            out[name] = m
        except Exception as e:
            st.session_state[f"load_err_s2_{name}"] = str(e)
    return out or None

XFORM = (
    T.Compose([T.Resize((224,224)), T.ToTensor(),
               T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    if TORCH_OK else None
)

def preprocess(pil_img):
    return XFORM(pil_img.convert("RGB")).unsqueeze(0)

# ═══════════════════════════════════════════════════════════════
# ══════════════════════  PAGES  ════════════════════════════════
# ═══════════════════════════════════════════════════════════════

# ── OVERVIEW ──────────────────────────────────────────────────
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🏥 Automated Skin Cancer Detection</div>
      <div class="hero-sub">
        BSc Final Year Project · University of Peshawar, Shaikh Zayed Islamic Centre<br>
        <strong style="color:#c7d2fe">Dual-Stage Deep Learning Pipeline</strong> &nbsp;·&nbsp;
        ISIC 2019 Dataset &nbsp;·&nbsp; 25,331 Images
      </div>
      <div class="hero-tags">
        <span class="tag">🔵 Stage 1 — SK1 Binary Triage</span>
        <span class="tag tag-amber">🟣 Final Stage 2 — Subtype Classification</span>
        <span class="tag tag-green">⚡ 5-Model Ensemble</span>
        <span class="tag tag-cyan">🧠 CNN + Vision Transformer</span>
        <span class="tag">🗺️ Grad-CAM XAI</span>
        <span class="tag">🎓 BSc CS 2022–2026</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    st.markdown('<div class="sec">⭐ Pipeline Headline Numbers</div>', unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    kpi_data = [
        (e1, "94.34%", "Stage 1 Accuracy",    "▲ SK1 Ensemble",        "kpi-val"),
        (e2, "98.63%", "Stage 1 AUC",         "Near-perfect triage",   "kpi-val"),
        (e3, "85.58%", "Stage 2 Accuracy",    "▲ Final Stage 2",       "kpi-val-amber"),
        (e4, "95.79%", "SCC Recall (Stage 2)","Best per-class recall",  "kpi-val-rose"),
    ]
    for col, val, lbl, delta, cls in kpi_data:
        col.markdown(f"""
        <div class="kpi">
          <div class="{cls}" style="font-size:2rem;font-weight:800;
            background:{'linear-gradient(135deg,#a5b4fc,#38bdf8)' if cls=='kpi-val' else
                        'linear-gradient(135deg,#fcd34d,#f59e0b)' if cls=='kpi-val-amber' else
                        'linear-gradient(135deg,#fda4af,#f43f5e)'};
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            font-family:'Space Grotesk',sans-serif;">
            {val}
          </div>
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown('<div class="sec">⚙️ Dual-Stage Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline">
      <div class="pipe-step"><div class="pipe-icon">🖼️</div><div class="pipe-title">Input Image</div>Dermoscopic lesion from ISIC 2019</div>
      <div class="pipe-step"><div class="pipe-icon">🔧</div><div class="pipe-title">Preprocessing</div>Hair removal · Lesion crop · Normalise</div>
      <div class="pipe-step"><div class="pipe-icon">🔵</div><div class="pipe-title">Stage 1 (SK1)</div>Benign vs Malignant ensemble triage</div>
      <div class="pipe-step"><div class="pipe-icon">🔀</div><div class="pipe-title">Router</div>Malignant → Stage 2 · Benign → cleared</div>
      <div class="pipe-step"><div class="pipe-icon">🟣</div><div class="pipe-title">Final Stage 2</div>MEL · BCC · SCC · AK subtype</div>
      <div class="pipe-step"><div class="pipe-icon">📋</div><div class="pipe-title">Report</div>Label + confidence + Grad-CAM</div>
    </div>
    """, unsafe_allow_html=True)

    # Key innovations
    st.markdown('<div class="sec">🧠 Key Technical Innovations</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    innovations = [
        (r1, "⚡ 5-Model Heterogeneous Ensemble",
              "ResNet50 · EfficientNet-B0 · MobileNetV2 · ConvNeXt-Tiny · Swin-Tiny combined via soft-voting. "
              "Diverse architectures reduce correlated errors; the ensemble outperforms every solo model."),
        (r2, "🎯 Dual Imbalance Strategy",
              "<b>Stage 1:</b> NV class undersampled 8,978→4,238 images. "
              "<b>Stage 2:</b> Focal Loss (γ=2) + MixUp (α=0.2) on the malignant subset — no data discarded."),
        (r3, "🗺️ Grad-CAM Explainability",
              "Gradient-weighted class activation maps confirm models focus on clinical lesion features "
              "(pigmentation networks, telangiectasia, ulceration borders) — not artefacts."),
    ]
    for col, t, b in innovations:
        col.markdown(f'<div class="card"><h4>{t}</h4><p>{b}</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    r4, r5, r6 = st.columns(3)
    for col, t, b in [
        (r4, "🏗️ From-Scratch Stage 2",
              "Stage 2 models use <b>random weight initialisation</b> (no ImageNet pre-training). "
              "Dermatological features have zero correlation with natural images — blank slate forces dermatology-grounded learning."),
        (r5, "🛡️ Clinical Safety Targets",
              "Ensemble specifically minimises False Negatives. Sensitivity targets: MEL ≥ 85%, SCC ≥ 95%. "
              "Stage 1 ensemble achieved <b>FN = 77</b> (down from 117 in solo MobileNetV2)."),
        (r6, "📊 Clinical Priority Score",
              "CPS = 0.40×MEL + 0.25×BCC + 0.25×SCC + 0.10×AK. "
              "Ensemble CPS = <b>0.888</b>, top score across all models, reflecting maximum protection for high-mortality subtypes."),
    ]:
        col.markdown(f'<div class="card"><h4>{t}</h4><p>{b}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="foot">🔬 SkinSense AI &nbsp;·&nbsp; University of Peshawar &nbsp;·&nbsp; Zohrouf Khattak & Huma Zeb &nbsp;·&nbsp; Supervisor: Dr Muhammad Ayaz<br><span style="color:#334155">⚠️ Research demo only — not for clinical use</span></div>', unsafe_allow_html=True)

# ── STAGE 1 (SK1) ─────────────────────────────────────────────
elif page == "Stage 1 (SK1)":
    st.markdown('<div style="font-family:Space Grotesk;font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#a5b4fc,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding-top:1rem;">🔵 Stage 1 — Binary Classification <span style="font-size:1rem;font-weight:600;color:#6366f1;">(SK1 Weights)</span></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;margin-bottom:1.2rem;">Classifying dermoscopic images as <b style="color:#10b981;">Benign</b> or <b style="color:#ef4444;">Malignant</b> using 5 ImageNet-pretrained models fine-tuned on the ISIC 2019 dataset. Weights stored in <code>SK1/</code>.</p>', unsafe_allow_html=True)

    # Gauges
    ens1 = S1["⭐ Ensemble"]
    g1, g2, g3, g4 = st.columns(4)
    g1.plotly_chart(gauge(ens1["Accuracy"],   "Accuracy",   "#6366f1"), use_container_width=True)
    g2.plotly_chart(gauge(ens1["Precision"],  "Precision",  "#8b5cf6"), use_container_width=True)
    g3.plotly_chart(gauge(ens1["Sensitivity"],"Sensitivity","#06b6d4"), use_container_width=True)
    g4.plotly_chart(gauge(ens1["AUC"]*100,    "AUC",        "#10b981"), use_container_width=True)

    # Confusion matrix
    st.markdown('<div class="sec">🔲 Ensemble Confusion Matrix</div>', unsafe_allow_html=True)
    cm_col, interp_col = st.columns([1, 1])
    with cm_col:
        fig_cm = go.Figure(go.Heatmap(
            z=[[2271, 128], [87, 1314]],
            x=["Predicted Benign", "Predicted Malignant"],
            y=["True Benign", "True Malignant"],
            colorscale=[[0,"#07090f"],[0.35,"#1e1b4b"],[1,"#6366f1"]],
            text=[["2271","128"],["87","1314"]],
            texttemplate="%{text}",
            textfont=dict(size=24, color="white", family="Space Grotesk"),
            showscale=False,
        ))
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8"),
                              height=280, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_cm, use_container_width=True)
    with interp_col:
        st.markdown("""
        <div class="card">
          <h4>📌 Clinical Interpretation (Test set: 3,801 images)</h4>
          <p>
            <b style="color:#10b981;">True Negatives: 2,267</b> — Benign correctly cleared<br><br>
            <b style="color:#f59e0b;">False Positives: 132</b> — Unnecessary referrals (safe side)<br><br>
            <b style="color:#ef4444;">False Negatives: 77</b> — Missed malignancies (critical!)<br><br>
            <b style="color:#6366f1;">True Positives: 1,314</b> — Cancers correctly detected<br><br>
            <small style="color:#64748b;">
              FNR = 5.50% &nbsp;|&nbsp; FPR = 5.50%<br>
              Ensemble FN (77) is 34% lower than best solo model MobileNetV2 (117 FN).
            </small>
          </p>
        </div>""", unsafe_allow_html=True)

    # Model comparison bar chart
    st.markdown('<div class="sec">📊 Model Comparison</div>', unsafe_allow_html=True)
    m_sel = st.selectbox("Select Metric", ["Accuracy","Precision","Sensitivity","F1","AUC"], key="s1_metric")
    st.plotly_chart(bar_chart(S1, m_sel, f"{m_sel} — SK1 Models"), use_container_width=True)

    # Full results table
    st.markdown('<div class="sec">📋 Complete Results Table</div>', unsafe_allow_html=True)
    df_s1 = pd.DataFrame([
        {"Model":k,"Accuracy":f"{v['Accuracy']:.2f}%","Precision":f"{v['Precision']:.2f}%",
         "Sensitivity":f"{v['Sensitivity']:.2f}%","F1":f"{v['F1']:.4f}","AUC":f"{v['AUC']:.4f}",
         "TP":v['TP'],"TN":v['TN'],"FP":v['FP'],"FN":v['FN']}
        for k,v in S1.items()
    ])
    st.dataframe(df_s1, use_container_width=True, hide_index=True)

    # Training curve images
    st.markdown('<div class="sec">📈 Training Convergence Curves</div>', unsafe_allow_html=True)
    tc_models = [("resnet50","ResNet50"),("efficientnet_b0","EfficientNet-B0"),
                 ("mobilenetv2","MobileNetV2"),("convnext_tiny","ConvNeXt-Tiny"),("swin_tiny","Swin-Tiny")]
    tca, tcb = st.columns(2)
    for i, (fname, mname) in enumerate(tc_models):
        path = os.path.join(SK1, f"{fname}_convergence_curves.png")
        img  = load_img(path)
        if img:
            (tca if i % 2 == 0 else tcb).image(img, caption=f"{mname} — Training & Validation Curves", use_container_width=True)

    # Clinical performance summary
    st.markdown('<div class="sec">🏥 Clinical Performance Summary</div>', unsafe_allow_html=True)
    cps_img = load_img(os.path.join(SK1, "clinical_performance_summary.png"))
    if cps_img:
        st.image(cps_img, caption="SK1 — Clinical Performance Summary", use_container_width=True)

    # ROC curves
    roc_img = load_img(os.path.join(SK1, "roc_curves_all_models_ensemble.png"))
    if roc_img:
        st.markdown('<div class="sec">📉 ROC Curves — All Models + Ensemble</div>', unsafe_allow_html=True)
        st.image(roc_img, caption="SK1 — ROC Curves (AUC shown per model)", use_container_width=True)

# ── FINAL STAGE 2 ─────────────────────────────────────────────
elif page == "Final Stage 2":
    st.markdown('<div style="font-family:Space Grotesk;font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#f9a8d4,#a78bfa 50%,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding-top:1rem;">🟣 Final Stage 2 — Malignant Subtype Classification</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;margin-bottom:1.2rem;">4-class classification of malignant lesions: <b style="color:#ef4444;">Melanoma (MEL)</b> · <b style="color:#f59e0b;">Basal Cell Carcinoma (BCC)</b> · <b style="color:#8b5cf6;">Squamous Cell Carcinoma (SCC)</b> · <b style="color:#06b6d4;">Actinic Keratosis (AK)</b>. Weights stored in <code>Final Stage 2/</code>.</p>', unsafe_allow_html=True)

    # Gauges
    e2 = S2["⭐ Ensemble"]
    g1, g2, g3, g4 = st.columns(4)
    g1.plotly_chart(gauge(e2["Accuracy"],    "Accuracy",    "#f59e0b"), use_container_width=True)
    g2.plotly_chart(gauge(e2["MacroF1"]*100, "Macro F1",    "#ef4444"), use_container_width=True)
    g3.plotly_chart(gauge(e2["MacroSens"],   "Macro Recall","#8b5cf6"), use_container_width=True)
    g4.plotly_chart(gauge(e2["AUC"]*100,     "Macro AUC",   "#06b6d4"), use_container_width=True)

    # Per-class recall
    st.markdown('<div class="sec">🎯 Per-Class Recall — Ensemble vs Individual Models</div>', unsafe_allow_html=True)
    recall_tab, radar_tab = st.tabs(["Bar Chart", "Radar Chart"])

    with recall_tab:
        fig_cls = go.Figure()
        clrs_m = ["#6366f1","#8b5cf6","#818cf8","#06b6d4","#38bdf8","#f59e0b"]
        for i,(mname,mc) in enumerate(zip(list(S2.keys()), clrs_m)):
            d = S2[mname]
            fig_cls.add_trace(go.Bar(
                name=mname, x=CLS_NAMES, y=[d["MEL"],d["BCC"],d["SCC"],d["AK"]],
                marker_color=mc, text=[f"{v:.1f}%" for v in [d["MEL"],d["BCC"],d["SCC"],d["AK"]]],
                textposition="outside", textfont=dict(color="#e2e8f0", size=10),
            ))
        fig_cls.update_layout(
            barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#94a3b8", showgrid=False),
            yaxis=dict(color="#94a3b8", gridcolor="#1e2a35", range=[60, 110]),
            height=360, margin=dict(t=10,b=20,l=44,r=20), font=dict(family="Inter"),
            legend=dict(bgcolor="rgba(15,22,35,.9)", bordercolor="#1e293b", borderwidth=1, font=dict(color="#e2e8f0", size=11)),
        )
        st.plotly_chart(fig_cls, use_container_width=True)

    with radar_tab:
        st.plotly_chart(radar_chart_s2(), use_container_width=True)

    # Clinical Priority Score
    st.markdown('<div class="sec">🏆 Clinical Priority Score (CPS)</div>', unsafe_allow_html=True)
    cps_col1, cps_col2 = st.columns([3,2])
    with cps_col1:
        cps_names = list(S2.keys())
        cps_vals  = [S2[m]["CPS"] * 100 for m in cps_names]
        fig_cps = go.Figure(go.Bar(
            x=cps_names, y=cps_vals,
            marker_color=["#6366f1","#8b5cf6","#818cf8","#06b6d4","#38bdf8","#f59e0b"],
            text=[f"{v:.2f}%" for v in cps_vals], textposition="outside",
            textfont=dict(color="#e2e8f0",size=11),
        ))
        fig_cps.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#94a3b8", showgrid=False, tickangle=-15),
            yaxis=dict(color="#94a3b8", gridcolor="#1e2a35", range=[80, 92]),
            height=290, margin=dict(t=20,b=60,l=44,r=20), showlegend=False, font=dict(family="Inter"),
        )
        st.plotly_chart(fig_cps, use_container_width=True)
    with cps_col2:
        st.markdown("""
        <div class="card">
          <h4>📐 Clinical Priority Score Formula</h4>
          <p>
            <code style="background:rgba(99,102,241,.12);padding:.15rem .5rem;border-radius:5px;color:#a5b4fc;">
              CPS = 0.40×MEL + 0.25×BCC + 0.25×SCC + 0.10×AK
            </code><br><br>
            Melanoma receives highest weight (40%) due to its mortality rate if missed. SCC and BCC each 25%.
            <br><br>
            <b style="color:#f59e0b;">Ensemble CPS = 0.8881</b> — best across all models.
          </p>
        </div>""", unsafe_allow_html=True)

    # Full results table
    st.markdown('<div class="sec">📋 Complete Stage 2 Results Table</div>', unsafe_allow_html=True)
    df_s2 = pd.DataFrame([
        {"Model":k,"Accuracy":f"{v['Accuracy']:.2f}%","Macro Prec.":f"{v['MacroPrec']:.2f}%",
         "Macro Sens.":f"{v['MacroSens']:.2f}%","Macro F1":f"{v['MacroF1']:.4f}",
         "AUC":f"{v['AUC']:.4f}","MEL":f"{v['MEL']:.2f}%","BCC":f"{v['BCC']:.2f}%",
         "SCC":f"{v['SCC']:.2f}%","AK":f"{v['AK']:.2f}%","CPS":f"{v['CPS']:.4f}"}
        for k,v in S2.items()
    ])
    st.dataframe(df_s2, use_container_width=True, hide_index=True)

    # Training configs card
    st.markdown('<div class="sec">⚙️ Stage 2 Training Configuration</div>', unsafe_allow_html=True)
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    cfg_col1.markdown("""
    <div class="card">
      <h4>🏗️ Weight Initialisation</h4>
      <p>All 5 models trained <b>from random weights</b> — no ImageNet pre-training.<br>
      Rationale: dermatological morphology has zero correlation with natural scene features.</p>
    </div>""", unsafe_allow_html=True)
    cfg_col2.markdown("""
    <div class="card">
      <h4>📉 Loss & Scheduler</h4>
      <p><b>Focal Loss</b> (γ=2) for rare-class focus (SCC, AK).<br>
      <b>CosineAnnealingLR</b> (T_max=50) for smooth from-scratch convergence.<br>
      LR = 1×10⁻³ (10× higher than Stage 1).</p>
    </div>""", unsafe_allow_html=True)
    cfg_col3.markdown("""
    <div class="card">
      <h4>🎲 MixUp + Augmentation</h4>
      <p><b>MixUp</b> (α=0.2) creates synthetic samples preventing SCC/AK overfitting.<br>
      Heavy augmentation: flips, rotation ±30°, colour jitter, random erasing.</p>
    </div>""", unsafe_allow_html=True)

    # Images
    st.markdown('<div class="sec">🖼️ Stage 2 Visualisations</div>', unsafe_allow_html=True)
    va, vb = st.columns(2)
    for i, (path, cap) in enumerate([
        (os.path.join(FS2,"confusion_matrices_1.png"), "Confusion Matrices — Part 1"),
        (os.path.join(FS2,"confusion_matrices_2.png"), "Confusion Matrices — Part 2"),
        (os.path.join(FS2,"roc_pr_curves.png"),        "ROC & PR Curves"),
        (os.path.join(FS2,"radar_chart.png"),           "Radar Chart — All Metrics"),
        (os.path.join(FS2,"training_convergence_grid.png"), "Training Convergence Grid"),
        (os.path.join(FS2,"ultimate_summary_table.png"),    "Summary Table"),
        (os.path.join(FS2,"grad_cam_AK.png"),  "Grad-CAM: Actinic Keratosis (AK)"),
        (os.path.join(FS2,"grad_cam_SCC.png"), "Grad-CAM: Squamous Cell Carcinoma (SCC)"),
    ]):
        img = load_img(path)
        if img:
            (va if i % 2 == 0 else vb).image(img, caption=cap, use_container_width=True)
            (va if i % 2 == 0 else vb).markdown("<br>", unsafe_allow_html=True)

# ── LIVE DEMO ──────────────────────────────────────────────────
elif page == "Live Demo":
    st.markdown('<div style="font-family:Space Grotesk;font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#34d399,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding-top:1rem;">🔮 Live Inference — Upload & Detect</div>', unsafe_allow_html=True)

    st.markdown('<div class="warn">⚠️ <b>Research Demo Only.</b> Predictions use real trained weights from the <code>SK1/</code> and <code>Final Stage 2/</code> folders, but must <em>not</em> be used for clinical diagnosis. Always consult a qualified dermatologist.</div>', unsafe_allow_html=True)

    if not TORCH_OK:
        st.markdown('<div class="err">❌ PyTorch not installed. Run: <code>pip install torch torchvision</code> then restart the app.</div>', unsafe_allow_html=True)
        st.stop()

    tab_s1, tab_s2, tab_full = st.tabs([
        "🔵 Stage 1 — Benign vs Malignant",
        "🟣 Stage 2 — Cancer Subtype",
        "🔗 Full Pipeline (S1 → S2)",
    ])

    # ── STAGE 1 TAB ──
    with tab_s1:
        st.markdown("##### Upload any skin lesion image — all 5 SK1 models will classify it as **Benign** or **Malignant**.")
        left, right = st.columns([1, 1], gap="large")
        with left:
            f1 = st.file_uploader("Choose image", type=["jpg","jpeg","png"], key="f1", label_visibility="collapsed")
            if f1:
                st.image(Image.open(f1), caption="Your image", use_container_width=True)
            else:
                st.markdown('<div class="up-hint">📸 Click or drag a dermoscopy image here<br><small>JPG · JPEG · PNG · Auto-resized to 224×224</small></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            btn1 = st.button("▶  Run SK1 Ensemble (Stage 1)", key="run1", disabled=(f1 is None))

        with right:
            if btn1 and f1:
                img1 = Image.open(f1)
                t1   = preprocess(img1)
                with st.spinner("Loading SK1 weights (first run only)…"):
                    mdls1 = load_s1_models()

                if not mdls1:
                    st.markdown('<div class="err">❌ Could not load Stage 1 weights — ensure <code>SK1/best_*.pth</code> files exist.</div>', unsafe_allow_html=True)
                else:
                    bar = st.progress(0, text="Starting inference…")
                    all_r1 = {}
                    for i, (name, m) in enumerate(mdls1.items()):
                        bar.progress(int((i+1)/len(mdls1)*88), text=f"Running {name}…")
                        with torch.no_grad():
                            p = torch.softmax(m(t1), 1).squeeze().numpy()
                        all_r1[name] = p
                    bar.progress(100, text="Averaging votes…"); time.sleep(0.12); bar.empty()

                    ens_p1 = np.mean(list(all_r1.values()), axis=0)
                    all_r1["⭐ Ensemble"] = ens_p1
                    is_mal = ens_p1[1] > 0.5
                    lbl    = "🔴 Malignant" if is_mal else "🟢 Benign"
                    lclr   = "#ef4444" if is_mal else "#10b981"
                    conf   = float(max(ens_p1)) * 100

                    st.markdown(f"""
                    <div class="pred-card">
                      <div style="font-size:.8rem;color:#94a3b8;margin-bottom:.4rem;">⭐ SK1 Ensemble Verdict</div>
                      <div class="pred-label" style="color:{lclr}">{lbl}</div>
                      <div style="font-size:.88rem;color:#94a3b8;">Confidence: {conf:.1f}%</div>
                      <div class="conf-bar">
                        <div class="conf-fill" style="width:{conf:.1f}%;background:linear-gradient(90deg,{lclr},{lclr}88);"></div>
                      </div>
                      <div style="font-size:.78rem;color:#64748b;">Benign {ens_p1[0]*100:.1f}% &nbsp;|&nbsp; Malignant {ens_p1[1]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br><div style='font-size:.87rem;font-weight:600;color:#e2e8f0;margin-bottom:.45rem;'>Per-model malignancy probability:</div>", unsafe_allow_html=True)
                    for name, probs in all_r1.items():
                        pm  = float(probs[1]) * 100
                        clr = "#f59e0b" if "Ensemble" in name else "#6366f1"
                        fw  = "700" if "Ensemble" in name else "400"
                        st.markdown(f"""
                        <div class="mbar">
                          <div class="mbar-name" style="color:{'#f59e0b' if 'Ensemble' in name else '#94a3b8'};font-weight:{fw};">{name}</div>
                          <div class="mbar-track"><div class="mbar-fill" style="width:{pm:.1f}%;background:linear-gradient(90deg,{clr},{clr}88);"></div></div>
                          <div class="mbar-val">{pm:.1f}%</div>
                        </div>""", unsafe_allow_html=True)

    # ── STAGE 2 TAB ──
    with tab_s2:
        st.markdown("##### Upload a **malignant** lesion — the Final Stage 2 ensemble identifies the cancer subtype.")
        left2, right2 = st.columns([1, 1], gap="large")
        with left2:
            f2 = st.file_uploader("Choose image", type=["jpg","jpeg","png"], key="f2", label_visibility="collapsed")
            if f2:
                st.image(Image.open(f2), caption="Your image", use_container_width=True)
            else:
                st.markdown('<div class="up-hint">📸 Drag a malignant lesion image here<br><small>Stage 2 classifies: MEL · BCC · SCC · AK</small></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            btn2 = st.button("▶  Run Final Stage 2 Ensemble", key="run2", disabled=(f2 is None))

        with right2:
            if btn2 and f2:
                img2 = Image.open(f2)
                t2   = preprocess(img2)
                with st.spinner("Loading Final Stage 2 weights (first run only)…"):
                    mdls2 = load_s2_models()

                if not mdls2:
                    st.markdown('<div class="err">❌ Could not load Stage 2 weights — ensure <code>Final Stage 2/*_stage2.pth</code> files exist.</div>', unsafe_allow_html=True)
                else:
                    bar2 = st.progress(0, text="Starting inference…")
                    all_r2 = {}
                    for i, (name, m) in enumerate(mdls2.items()):
                        bar2.progress(int((i+1)/len(mdls2)*88), text=f"Running {name}…")
                        with torch.no_grad():
                            p = torch.softmax(m(t2), 1).squeeze().numpy()
                        all_r2[name] = p
                    bar2.progress(100, text="Averaging votes…"); time.sleep(0.12); bar2.empty()

                    ens_p2 = np.mean(list(all_r2.values()), axis=0)
                    all_r2["⭐ Ensemble"] = ens_p2
                    top_idx  = int(np.argmax(ens_p2))
                    top_conf = float(ens_p2[top_idx]) * 100
                    top_clr  = CLS_COLORS[top_idx]

                    st.markdown(f"""
                    <div class="pred-card">
                      <div style="font-size:.8rem;color:#94a3b8;margin-bottom:.4rem;">⭐ Final Stage 2 Ensemble Verdict</div>
                      <div class="pred-label" style="color:{top_clr};font-size:1.3rem;">{CLS_NAMES[top_idx]}</div>
                      <div style="font-size:.88rem;color:#94a3b8;">Confidence: {top_conf:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

                    # Probability bars per class
                    st.markdown("<br>", unsafe_allow_html=True)
                    fig_p2 = go.Figure(go.Bar(
                        x=CLS_NAMES, y=[float(p)*100 for p in ens_p2],
                        marker_color=CLS_COLORS,
                        text=[f"{float(p)*100:.1f}%" for p in ens_p2],
                        textposition="outside", textfont=dict(color="#e2e8f0"),
                    ))
                    fig_p2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(range=[0,110], gridcolor="#1e2a35", color="#94a3b8"),
                        xaxis=dict(color="#94a3b8", tickangle=-10, showgrid=False),
                        height=270, margin=dict(t=10,b=60,l=40,r=10), showlegend=False,
                        font=dict(family="Inter"),
                        title=dict(text="Ensemble subtype probabilities",font=dict(color="#94a3b8",size=12),x=0),
                    )
                    st.plotly_chart(fig_p2, use_container_width=True)

                    st.markdown("<div style='font-size:.87rem;font-weight:600;color:#e2e8f0;margin-bottom:.4rem;'>Per-model prediction:</div>", unsafe_allow_html=True)
                    for name, probs in all_r2.items():
                        pi    = int(np.argmax(probs))
                        pclr  = CLS_COLORS[pi]
                        pconf = float(probs[pi]) * 100
                        is_e  = "Ensemble" in name
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:.7rem;margin:.28rem 0;
                             {'background:rgba(245,158,11,.08);border-radius:7px;padding:.25rem .5rem;' if is_e else ''}">
                          <div style="width:145px;font-size:.8rem;color:{'#f59e0b' if is_e else '#94a3b8'};font-weight:{'700' if is_e else '400'};flex-shrink:0;">{name}</div>
                          <div style="flex:1;font-size:.8rem;color:{pclr};font-weight:600;">{CLS_NAMES[pi]}</div>
                          <div style="width:44px;font-size:.8rem;color:#94a3b8;text-align:right;">{pconf:.0f}%</div>
                        </div>""", unsafe_allow_html=True)

    # ── FULL PIPELINE TAB ──
    with tab_full:
        st.markdown("##### Upload an image — it runs through **Stage 1 first**, then if malignant → **Stage 2 automatically**.")
        st.markdown('<div class="info">ℹ️ This tab simulates the complete clinical pipeline: triage first, then detailed subtype if needed.</div>', unsafe_allow_html=True)
        left3, right3 = st.columns([1, 1], gap="large")
        with left3:
            f3 = st.file_uploader("Choose image", type=["jpg","jpeg","png"], key="f3", label_visibility="collapsed")
            if f3:
                st.image(Image.open(f3), caption="Your image", use_container_width=True)
            else:
                st.markdown('<div class="up-hint">📸 Upload any skin lesion<br><small>Auto-routed through both stages if malignant</small></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            btn3 = st.button("▶  Run Full Pipeline (Stage 1 → Stage 2)", key="run3", disabled=(f3 is None))

        with right3:
            if btn3 and f3:
                img3 = Image.open(f3)
                t3   = preprocess(img3)

                # Stage 1
                with st.spinner("Stage 1 — Loading SK1 weights…"):
                    mdls1_p = load_s1_models()
                if not mdls1_p:
                    st.markdown('<div class="err">❌ SK1 weights not found.</div>', unsafe_allow_html=True)
                    st.stop()

                bar3 = st.progress(0, text="Stage 1 inference…")
                r1_all = {}
                for i, (n, m) in enumerate(mdls1_p.items()):
                    bar3.progress(int((i+1)/len(mdls1_p)*50), text=f"S1: {n}…")
                    with torch.no_grad():
                        p = torch.softmax(m(t3), 1).squeeze().numpy()
                    r1_all[n] = p
                bar3.progress(50, text="Stage 1 done — checking result…"); time.sleep(0.1)

                ens_p3 = np.mean(list(r1_all.values()), axis=0)
                is_mal3 = ens_p3[1] > 0.5
                s1_conf = float(max(ens_p3)) * 100
                s1_lbl  = "🔴 Malignant" if is_mal3 else "🟢 Benign"
                s1_clr  = "#ef4444" if is_mal3 else "#10b981"

                st.markdown(f"""
                <div style="border:1px solid {s1_clr}33;border-radius:12px;padding:1rem 1.2rem;background:rgba(15,22,35,.9);margin-bottom:.8rem;">
                  <div style="font-size:.75rem;color:#94a3b8;margin-bottom:.25rem;">Stage 1 (SK1) Result</div>
                  <div style="font-size:1.35rem;font-weight:800;color:{s1_clr};font-family:'Space Grotesk',sans-serif;">{s1_lbl}</div>
                  <div style="font-size:.82rem;color:#94a3b8;margin-top:.25rem;">Confidence: {s1_conf:.1f}% &nbsp;|&nbsp; Benign {ens_p3[0]*100:.1f}% · Malignant {ens_p3[1]*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

                if not is_mal3:
                    bar3.progress(100, text="Cleared — no Stage 2 needed."); time.sleep(0.15); bar3.empty()
                    st.markdown('<div style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);border-radius:11px;padding:1rem;text-align:center;color:#6ee7b7;font-weight:700;">✅ Routed to: CLEARED (Benign)<br><small style="font-weight:400;color:#94a3b8;">Stage 2 not triggered — lesion classified as benign by SK1 ensemble.</small></div>', unsafe_allow_html=True)
                else:
                    bar3.progress(55, text="Malignant detected — routing to Stage 2…"); time.sleep(0.12)
                    with st.spinner("Stage 2 — Loading Final Stage 2 weights…"):
                        mdls2_p = load_s2_models()
                    if not mdls2_p:
                        bar3.empty()
                        st.markdown('<div class="err">❌ Final Stage 2 weights not found.</div>', unsafe_allow_html=True)
                        st.stop()

                    r2_all = {}
                    for i, (n, m) in enumerate(mdls2_p.items()):
                        bar3.progress(55 + int((i+1)/len(mdls2_p)*43), text=f"S2: {n}…")
                        with torch.no_grad():
                            p = torch.softmax(m(t3), 1).squeeze().numpy()
                        r2_all[n] = p
                    bar3.progress(100, text="Done!"); time.sleep(0.15); bar3.empty()

                    ens_p4   = np.mean(list(r2_all.values()), axis=0)
                    top4     = int(np.argmax(ens_p4))
                    top4_c   = float(ens_p4[top4]) * 100

                    st.markdown(f"""
                    <div class="pred-card" style="border-color:{CLS_COLORS[top4]}44;">
                      <div style="font-size:.8rem;color:#94a3b8;margin-bottom:.3rem;">⭐ Final Diagnosis — Stage 2 Ensemble</div>
                      <div class="pred-label" style="color:{CLS_COLORS[top4]};font-size:1.35rem;">{CLS_NAMES[top4]}</div>
                      <div style="font-size:.88rem;color:#94a3b8;margin-bottom:.7rem;">Confidence: {top4_c:.1f}%</div>
                      <div style="display:flex;justify-content:center;gap:.6rem;flex-wrap:wrap;">
                        {''.join(f'<div style="background:{CLS_COLORS[i]}22;border:1px solid {CLS_COLORS[i]}44;border-radius:8px;padding:.3rem .7rem;font-size:.78rem;color:{CLS_COLORS[i]};font-weight:600;">{CLS_ABBR[i]}: {float(ens_p4[i])*100:.1f}%</div>' for i in range(4))}
                      </div>
                    </div>""", unsafe_allow_html=True)

# ── GALLERY ────────────────────────────────────────────────────
elif page == "Gallery":
    st.markdown('<div style="font-family:Space Grotesk;font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#f9a8d4,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding-top:1rem;">🖼️ Results Gallery</div>', unsafe_allow_html=True)

    t_s1, t_s2, t_prep, t_cam = st.tabs(["Stage 1 (SK1)", "Final Stage 2", "Preprocessing", "Grad-CAM"])

    with t_s1:
        c1, c2 = st.columns(2); ci = 0
        for path, cap in [
            (os.path.join(SK1,"roc_curves_all_models_ensemble.png"), "ROC Curves — All Models + Ensemble"),
            (os.path.join(SK1,"clinical_performance_summary.png"),   "Clinical Performance Summary"),
            (os.path.join(SK1,"confusion_matrices_all_models.png"),  "Confusion Matrices — All Models"),
            (os.path.join(SK1,"ablation_study_results.png"),         "Ablation Study Results"),
            (os.path.join(SK1,"model_comparison_chart.png"),         "Model Comparison Chart"),
            (os.path.join(SK1,"02_split_distribution.png"),          "Dataset Split Distribution"),
            (os.path.join(SK1,"03_undersampling_effect.png"),        "Undersampling Effect (NV Class)"),
            (os.path.join(SK1,"04_full_class_distribution.png"),     "Full Class Distribution"),
        ]:
            img = load_img(path)
            if img:
                (c1 if ci % 2 == 0 else c2).image(img, caption=cap, use_container_width=True)
                (c1 if ci % 2 == 0 else c2).markdown("<br>", unsafe_allow_html=True)
                ci += 1

    with t_s2:
        c1, c2 = st.columns(2); ci = 0
        for path, cap in [
            (os.path.join(FS2,"roc_pr_curves.png"),              "ROC & PR Curves"),
            (os.path.join(FS2,"radar_chart.png"),                  "Model Radar Chart"),
            (os.path.join(FS2,"confusion_matrices_1.png"),         "Confusion Matrices Part 1"),
            (os.path.join(FS2,"confusion_matrices_2.png"),         "Confusion Matrices Part 2"),
            (os.path.join(FS2,"training_convergence_grid.png"),    "Training Convergence Grid"),
            (os.path.join(FS2,"ultimate_summary_table.png"),       "Ultimate Summary Table"),
            (os.path.join(FS2,"clinical_priority_scores.png"),     "Clinical Priority Scores"),
            (os.path.join(FS2,"raw_distribution.png"),             "Raw Class Distribution"),
            (os.path.join(FS2,"undersampling_effect.png"),         "Stage 2 Imbalance View"),
        ]:
            img = load_img(path)
            if img:
                (c1 if ci % 2 == 0 else c2).image(img, caption=cap, use_container_width=True)
                (c1 if ci % 2 == 0 else c2).markdown("<br>", unsafe_allow_html=True)
                ci += 1

    with t_prep:
        img_pre = load_img(os.path.join(FS2, "clean_preprocessing.png"))
        img_aug = load_img(os.path.join(FS2, "heavy_augmentation.png"))
        img_aug_sk1 = load_img(os.path.join(SK1, "augmentation_visualization.png"))
        img_7step   = load_img(os.path.join(SK1, "preprocessing_7step_thesis.png"))
        for img, cap in [(img_7step,"7-Step Preprocessing Pipeline"),(img_pre,"Clean Preprocessing Pipeline"),
                         (img_aug_sk1,"Stage 1 Augmentation"),(img_aug,"Stage 2 — MixUp Augmentation")]:
            if img:
                st.image(img, caption=cap, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

    with t_cam:
        cam1 = load_img(os.path.join(FS2,"grad_cam_AK.png"))
        cam2 = load_img(os.path.join(FS2,"grad_cam_SCC.png"))
        cc1, cc2 = st.columns(2)
        if cam1: cc1.image(cam1, caption="Grad-CAM: Actinic Keratosis (AK)", use_container_width=True)
        if cam2: cc2.image(cam2, caption="Grad-CAM: Squamous Cell Carcinoma (SCC)", use_container_width=True)
        st.markdown("""
        <div class="card" style="margin-top:1rem;">
          <h4>🗺️ Grad-CAM Interpretation</h4>
          <p>
            Gradient-weighted Class Activation Maps highlight which image regions were most influential for the model's prediction.
            <b>AK heatmaps</b> concentrate on scaly keratotic surface texture.
            <b>SCC heatmaps</b> focus on central keratotic plugs, ulceration, and irregular erythematous borders —
            classic clinical hallmarks. Models correctly ignore background artefacts, rulers, and normal skin,
            demonstrating genuine dermatological reasoning.
          </p>
        </div>""", unsafe_allow_html=True)

# ── ABOUT ──────────────────────────────────────────────────────
elif page == "About":
    st.markdown('<div style="font-family:Space Grotesk;font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#a5b4fc,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding-top:1rem;">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">👩‍💻 Project Team</div>', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns(3)
    for col, emoji, name, role, detail in [
        (tc1,"👤","Zohrouf Khattak","Student Researcher","Roll No: 1000\nBSc Computer Science 2022–2026"),
        (tc2,"👤","Huma Zeb","Student Researcher","Roll No: 883\nBSc Computer Science 2022–2026"),
        (tc3,"🎓","Dr Muhammad Ayaz","Project Supervisor","University of Peshawar\nShaikh Zayed Islamic Centre"),
    ]:
        col.markdown(f'<div class="card" style="text-align:center"><div style="font-size:2rem;margin-bottom:.5rem">{emoji}</div><h4 style="color:#e2e8f0;font-size:1rem">{name}</h4><div style="font-size:.76rem;color:#6366f1;font-weight:700;margin-bottom:.5rem;text-transform:uppercase;letter-spacing:.04em">{role}</div><p style="white-space:pre-line">{detail}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">🏗️ Model Architectures</div>', unsafe_allow_html=True)
    archs = [
        ("ResNet50", "CNN — 50 layers", "Skip connections solve vanishing gradients. Well-understood medical imaging baseline. Stage 1: fine-tuned with Dropout(0.5) head. Stage 2: random init, plain linear head."),
        ("EfficientNet-B0", "Compound Scaling CNN", "Balances depth, width and resolution via a compound coefficient. Best accuracy-per-FLOP. Only ~5.3M params — top mobile candidate."),
        ("MobileNetV2", "Lightweight Depthwise CNN", "Inverted residual blocks for fast CPU/mobile inference. Depthwise separable convolutions reduce computation significantly."),
        ("ConvNeXt-Tiny", "Modern Pure CNN", "Rebuilt from scratch to match Vision Transformer strengths while retaining CNN efficiency. Uses LayerNorm and GELU."),
        ("Swin-Tiny", "Shifted-Window Transformer", "Treats 16×16 image patches as tokens with hierarchical self-attention. Captures global lesion structure — particularly effective on AK."),
    ]
    for name, arch, desc in archs:
        with st.expander(f"**{name}** — {arch}"):
            st.markdown(f"<p style='color:#94a3b8;line-height:1.8;font-size:.88rem'>{desc}</p>", unsafe_allow_html=True)

    st.markdown('<div class="sec">📂 Folder Structure</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
      <h4>Weight Files Used By This Demo</h4>
      <p>
        <b style="color:#a5b4fc;">SK1/</b> — Stage 1 (Binary Triage) trained weights:<br>
        <code>best_resnet50.pth · best_efficientnet-b0.pth · best_mobilenetv2.pth · best_convnext-tiny.pth · best_swin-tiny.pth</code>
        <br><br>
        <b style="color:#f9a8d4;">Final Stage 2/</b> — Stage 2 (Subtype Classification) trained weights:<br>
        <code>ResNet50_stage2.pth · EfficientNet-B0_stage2.pth · MobileNetV2_stage2.pth · ConvNeXt-Tiny_stage2.pth · Swin-Tiny_stage2.pth</code>
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">⚙️ Tech Stack</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.columns(5)
    for col, n, d in [
        (t1, "🔥 PyTorch", "Deep learning framework"),
        (t2, "👁️ torchvision", "Model zoo & transforms"),
        (t3, "📊 Streamlit", "Web demo framework"),
        (t4, "📈 Plotly", "Interactive charts"),
        (t5, "🗺️ Grad-CAM", "XAI heatmaps"),
    ]:
        col.markdown(f'<div class="card" style="text-align:center"><h4>{n}</h4><p>{d}</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="foot">
      🔬 SkinSense AI &nbsp;·&nbsp; University of Peshawar &nbsp;·&nbsp;
      Zohrouf Khattak & Huma Zeb &nbsp;·&nbsp; Supervisor: Dr Muhammad Ayaz<br>
      <span style="color:#334155">⚠️ Research & educational purposes only — not for clinical use</span>
    </div>""", unsafe_allow_html=True)
