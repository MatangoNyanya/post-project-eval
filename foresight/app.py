from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yaml
except ImportError:
    yaml = None

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import COUNTRY_REGION, SUBSECTORS
from logic import (
    calculate_mock_risk,
    generate_mock_shap,
    generate_sub_ratings,
    select_lesson_data,
    select_similar_projects,
)

CONFIG_PATH = ROOT_DIR / "config.yaml"

DEFAULT_CONFIG: dict = {
    "ui": {
        "budget":   {"default": 8.5,  "min": 0.1,  "max": 500.0, "step": 0.1},
        "duration": {"default": 36,   "min": 6,    "max": 120,   "step": 1},
        "default_country": "ラオス",
        "default_sector":  "運輸交通",
        "default_scheme":  "技術協力",
    }
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
[data-testid="stAppViewContainer"] { background: #f8fafc; color: #0f172a; }
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * { color: #334155 !important; }

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 32px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #003A78, #0057B8, #2563eb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1.0rem;
    color: #64748b;
    margin: 0;
    font-weight: 400;
}

/* ── Step indicator ── */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 28px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    flex: 1;
}
.step-circle {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 700;
    background: #e2e8f0; color: #94a3b8;
    flex-shrink: 0;
}
.step-circle.active { background: #003A78; color: #ffffff; }
.step-circle.done   { background: #059669; color: #ffffff; }
.step-label { font-size: 0.85rem; font-weight: 600; color: #94a3b8; }
.step-label.active  { color: #0f172a; }
.step-connector { width: 40px; height: 2px; background: #e2e8f0; margin: 0 8px; flex-shrink: 0; }
.step-connector.done { background: #059669; }

/* ── Section header ── */
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #1e293b;
    margin: 28px 0 14px 0; padding-bottom: 10px;
    border-bottom: 2px solid #e2e8f0;
    letter-spacing: -0.01em;
}

/* ── Verdict ── */
.verdict-good    { background:#ecfdf5; border:1px solid #6ee7b7; color:#065f46;
                   border-radius:12px; padding:16px 24px; font-size:1.05rem; font-weight:600; }
.verdict-warning { background:#fffbeb; border:1px solid #fcd34d; color:#78350f;
                   border-radius:12px; padding:16px 24px; font-size:1.05rem; font-weight:600; }
.verdict-danger  { background:#fef2f2; border:1px solid #fca5a5; color:#7f1d1d;
                   border-radius:12px; padding:16px 24px; font-size:1.05rem; font-weight:600; }

/* ── Rating cards ── */
.rating-grid { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }
.rating-card {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
    padding:16px 20px; text-align:center; flex:1; min-width:100px;
    box-shadow:0 1px 2px rgba(0,0,0,0.04);
}
.rating-name  { font-size:0.78rem; color:#64748b; font-weight:600;
                text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px; }
.rating-value { font-size:1.8rem; font-weight:800; }
.rating-label { font-size:0.78rem; color:#94a3b8; margin-top:4px; }
.r3 { color:#059669; } .r2 { color:#d97706; } .r1 { color:#dc2626; }

/* ── SHAP explanations ── */
.shap-item {
    display:flex; align-items:flex-start; gap:14px;
    background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
    padding:14px 18px; margin-bottom:10px;
    box-shadow:0 1px 2px rgba(0,0,0,0.03);
}
.shap-icon  { font-size:1.4rem; flex-shrink:0; }
.shap-fname { font-size:0.9rem; font-weight:700; color:#1e293b; margin-bottom:4px; }
.shap-exp   { font-size:0.85rem; color:#64748b; line-height:1.5; }
.badge-risk { background:#fff7ed; color:#c2410c; border:1px solid #fed7aa;
              border-radius:20px; padding:3px 10px; font-size:0.75rem; font-weight:700;
              white-space:nowrap; margin-left:auto; flex-shrink:0; }
.badge-safe { background:#ecfdf5; color:#15803d; border:1px solid #bbf7d0;
              border-radius:20px; padding:3px 10px; font-size:0.75rem; font-weight:700;
              white-space:nowrap; margin-left:auto; flex-shrink:0; }

/* ── Similar projects table ── */
.eval-A { background:#d1fae5; color:#065f46; border-radius:6px; padding:2px 8px; font-weight:700; }
.eval-B { background:#dbeafe; color:#1e40af; border-radius:6px; padding:2px 8px; font-weight:700; }
.eval-C { background:#fef3c7; color:#92400e; border-radius:6px; padding:2px 8px; font-weight:700; }
.eval-D { background:#fee2e2; color:#991b1b; border-radius:6px; padding:2px 8px; font-weight:700; }

/* ── Lesson panel ── */
.lesson-summary {
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:16px;
    padding:28px 32px; margin-top:24px;
}
.lesson-summary-title { font-size:1.1rem; font-weight:700; color:#1e3a8a; margin-bottom:16px; }
.summary-point {
    display:flex; gap:14px; align-items:flex-start;
    margin-bottom:14px; padding:14px 16px;
    background:#ffffff; border-radius:10px; border:1px solid #bfdbfe;
}
.summary-num {
    width:28px; height:28px; border-radius:50%; background:#003A78; color:#fff;
    font-size:0.85rem; font-weight:800; display:flex; align-items:center;
    justify-content:center; flex-shrink:0;
}
.summary-text    { font-size:0.9rem; color:#1e293b; font-weight:500; }
.summary-related { font-size:0.78rem; color:#64748b; margin-top:4px; }

/* ── Info banner ── */
.info-banner {
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:16px;
    padding:48px; text-align:center; color:#1e3a8a; font-size:1.05rem;
    margin-top:24px; font-weight:500;
}

/* ── Form submit ── */
div[data-testid="stFormSubmitButton"] > button {
    background: #003A78 !important; color: #ffffff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 12px !important; width: 100% !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #002a58 !important;
}
div[data-testid="stFormSubmitButton"] > button p,
div[data-testid="stFormSubmitButton"] > button span { color: #ffffff !important; }

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label {
    font-size: 0.85rem !important; font-weight: 600 !important;
    color: #475569 !important; text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
</style>
"""


# ──────────────────────────────────────────────────────────────
# 設定ロード
# ──────────────────────────────────────────────────────────────

def _deep_update(base: dict, update: dict) -> dict:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    if not path.exists() or yaml is None:
        return cfg
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _deep_update(cfg, loaded)
    except Exception:
        return cfg


APP_CONFIG = _load_config()
UI_CFG = APP_CONFIG.get("ui", {})


# ──────────────────────────────────────────────────────────────
# Plotly チャート
# ──────────────────────────────────────────────────────────────

def _build_gauge(risk_score: float) -> go.Figure:
    if risk_score < 40:
        color = "#2E7D32"
    elif risk_score < 60:
        color = "#FF6D00"
    else:
        color = "#D32F2F"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": "%", "font": {"size": 48, "color": color, "family": "Inter, sans-serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  40], "color": "#C8E6C9"},
                {"range": [40, 60], "color": "#FFE0B2"},
                {"range": [60, 100], "color": "#FFCDD2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": risk_score,
            },
        },
        title={"text": "要注意確率", "font": {"size": 13, "color": "#64748b"}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#ffffff",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def _build_shap_chart(shap_factors: list[dict]) -> go.Figure:
    labels = [f"{f['icon']} {f['factor']}" for f in shap_factors]
    values = [f["value"] for f in shap_factors]
    colors = ["#FF6D00" if f["direction"] == "risk" else "#2E7D32" for f in shap_factors]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{'+' if v >= 0 else ''}{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, family="Inter, sans-serif"),
    ))
    fig.update_layout(
        height=max(220, len(shap_factors) * 46),
        xaxis=dict(
            title="SHAP値",
            tickfont=dict(size=11, color="#64748b"),
            gridcolor="#f1f5f9",
            zerolinecolor="#cbd5e1",
        ),
        yaxis=dict(tickfont=dict(size=11, color="#374151")),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        margin=dict(l=10, r=60, t=10, b=30),
        font={"family": "Inter, sans-serif"},
    )
    return fig


# ──────────────────────────────────────────────────────────────
# UI パーツ
# ──────────────────────────────────────────────────────────────

def _render_step_indicator(current_step: int) -> None:
    steps = ["入力", "リスク予測", "教訓・対策"]

    def circle_class(i: int) -> str:
        if i + 1 < current_step:
            return "done"
        if i + 1 == current_step:
            return "active"
        return ""

    def label_class(i: int) -> str:
        return "active" if i + 1 <= current_step else ""

    def connector_class(i: int) -> str:
        return "done" if i + 1 < current_step else ""

    circle_symbols = ["✓" if i + 1 < current_step else str(i + 1) for i in range(3)]

    parts = []
    for i, step_name in enumerate(steps):
        parts.append(
            f'<div class="step-item">'
            f'  <div class="step-circle {circle_class(i)}">{circle_symbols[i]}</div>'
            f'  <span class="step-label {label_class(i)}">{step_name}</span>'
            f'</div>'
        )
        if i < len(steps) - 1:
            parts.append(f'<div class="step-connector {connector_class(i)}"></div>')

    st.markdown(
        f'<div class="step-indicator">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def _render_rating_cards(ratings: list[dict]) -> None:
    cls_map = {"③": "r3", "②": "r2", "①": "r1"}
    cards = "".join(
        f'<div class="rating-card">'
        f'  <div class="rating-name">{r["name"]}</div>'
        f'  <div class="rating-value {cls_map.get(r["rating"], "")}">{r["rating"]}</div>'
        f'  <div class="rating-label">{r["label"]}</div>'
        f'</div>'
        for r in ratings
    )
    st.markdown(f'<div class="rating-grid">{cards}</div>', unsafe_allow_html=True)


def _render_shap_explanations(shap_factors: list[dict]) -> None:
    for f in shap_factors:
        badge_cls = "badge-risk" if f["direction"] == "risk" else "badge-safe"
        badge_txt = f"▲ {f['value']:+.2f} リスク" if f["direction"] == "risk" else f"▼ {f['value']:+.2f} 低減"
        st.markdown(
            f'<div class="shap-item">'
            f'  <div class="shap-icon">{f["icon"]}</div>'
            f'  <div style="flex:1">'
            f'    <div class="shap-fname">{f["factor"]}</div>'
            f'    <div class="shap-exp">{f["explanation"]}</div>'
            f'  </div>'
            f'  <span class="{badge_cls}">{badge_txt}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_similar_projects(projects: list[dict]) -> None:
    df = pd.DataFrame(projects)
    counts = df["総合評価"].value_counts().to_dict()
    total = len(projects)

    chips = " ".join(
        f'<span class="eval-{g}">{g}={counts.get(g, 0)}件 ({round(counts.get(g, 0)/total*100)}%)</span>'
        for g in ["A", "B", "C", "D"]
    )
    st.markdown(
        f'<p style="font-size:0.8rem;color:#6b7280;margin-bottom:8px">'
        f'類似案件{total}件中：{chips}</p>',
        unsafe_allow_html=True,
    )

    display_df = df.copy()
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "事業名":   st.column_config.TextColumn("事業名",   width="large"),
            "評価年":   st.column_config.NumberColumn("評価年",  format="%d"),
            "セクター": st.column_config.TextColumn("セクター"),
            "総合評価": st.column_config.TextColumn("総合評価"),
            "効率性":   st.column_config.TextColumn("効率性"),
            "持続性":   st.column_config.TextColumn("持続性"),
        },
    )


def _render_lesson_panels(lesson_data: dict) -> None:
    for rf in lesson_data["risk_factors"]:
        header = (
            f"{rf['factor_icon']} **{rf['factor_name']}**  "
            f"— SHAP {rf['shap_value']} | 類似事業 {rf['similar_count']}件"
        )
        with st.expander(header, expanded=(rf["shap_rank"] == 1)):
            for li, lesson in enumerate(rf["key_lessons"]):
                st.markdown(f"**🔑 教訓 {rf['shap_rank']}-{li+1}：{lesson['title']}**")
                st.markdown(
                    f'<div style="background:#f8fafc;border-left:4px solid #003A78;'
                    f'padding:14px 18px;border-radius:0 8px 8px 0;'
                    f'font-size:0.92rem;line-height:1.65;color:#334155;margin-bottom:10px">'
                    f'{lesson["text"]}</div>',
                    unsafe_allow_html=True,
                )
                source_tags = " ".join(
                    f'<span style="background:#f1f5f9;border:1px solid #e2e8f0;'
                    f'border-radius:20px;padding:3px 10px;font-size:0.75rem;color:#475569">'
                    f'📎 {s["name"]}（{s["year"]}）{s["page"]}</span>'
                    for s in lesson["sources"]
                )
                st.markdown(source_tags, unsafe_allow_html=True)

                if li < len(rf["key_lessons"]) - 1:
                    st.markdown("---")

            st.markdown("**✅ 成功事業で見られた対策パターン**")
            for pat in rf["success_patterns"]:
                st.markdown(f"- {pat}")

            if "statistics" in rf:
                st.markdown(f"**📊 統計的補足：{rf['statistics']['description']}**")
                for d in rf["statistics"]["data"]:
                    st.progress(d["value"] / 100, text=f"{d['label']}：{d['value']}% ({d['detail']})")


def _render_summary(summary: dict) -> None:
    points_html = "".join(
        f'<div class="summary-point">'
        f'  <div class="summary-num">{p["number"]}</div>'
        f'  <div>'
        f'    <div class="summary-text">{p["text"]}</div>'
        f'    <div class="summary-related">💬 関連要因: {p["related_factor"]}</div>'
        f'  </div>'
        f'</div>'
        for p in summary["points"]
    )
    st.markdown(
        f'<div class="lesson-summary">'
        f'  <div class="lesson-summary-title">{summary["title"]}</div>'
        f'  {points_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# ページ設定
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Foresight | リスクスクリーニング",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# セッション初期化
for key, val in [("step", 1), ("input_data", {}), ("risk_score", None), ("shap_factors", [])]:
    if key not in st.session_state:
        st.session_state[key] = val

# ──────────────────────────────────────────────────────────────
# ヒーローヘッダー
# ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-header">'
    '  <div class="hero-title">🧭 Foresight</div>'
    '  <div class="hero-subtitle">'
    '    JICA 事後評価リスクスクリーニングツール &nbsp;|&nbsp;'
    '    プロジェクト情報を入力すると、過去の類似案件から評価リスクと教訓を提示します。'
    '  </div>'
    '</div>',
    unsafe_allow_html=True,
)

# ステップインジケーター
_render_step_indicator(st.session_state.step)

# ──────────────────────────────────────────────────────────────
# サイドバー
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Foresight")
    st.markdown("---")
    st.markdown(
        "**JICA事後評価リスクスクリーニングツール**\n\n"
        "プロジェクトの基本情報を入力すると、2,000件超の過去評価データから "
        "類似案件を参照し、評価リスクと対策教訓を提示します。\n\n"
        "> ⚠️ 本ツールはプロトタイプです。予測値はダミーロジックに基づきます。"
    )
    st.markdown("---")
    st.markdown("**現在のステップ**")
    step_names = {1: "① 入力", 2: "② リスク予測", 3: "③ 教訓・対策"}
    st.info(step_names.get(st.session_state.step, ""))

    if st.session_state.step > 1:
        st.markdown("---")
        if st.button("🔄 最初からやり直す", use_container_width=True):
            st.session_state.step = 1
            st.session_state.input_data = {}
            st.session_state.risk_score = None
            st.session_state.shap_factors = []
            st.rerun()

# ──────────────────────────────────────────────────────────────
# STEP 1: 入力フォーム
# ──────────────────────────────────────────────────────────────
if st.session_state.step == 1:
    st.markdown('<div class="section-header">📋 プロジェクト情報の入力</div>', unsafe_allow_html=True)

    budget_cfg   = UI_CFG.get("budget", {})
    duration_cfg = UI_CFG.get("duration", {})
    countries    = list(COUNTRY_REGION.keys())
    sectors      = list(SUBSECTORS.keys())
    schemes      = ["技術協力", "無償資金協力", "有償資金協力"]

    default_country = UI_CFG.get("default_country", countries[0])
    default_sector  = UI_CFG.get("default_sector",  sectors[0])
    default_scheme  = UI_CFG.get("default_scheme",  schemes[0])

    default_country_idx = countries.index(default_country) if default_country in countries else 0
    default_sector_idx  = sectors.index(default_sector)    if default_sector  in sectors  else 0
    default_scheme_idx  = schemes.index(default_scheme)    if default_scheme  in schemes  else 0

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            country = st.selectbox("対象国", countries, index=default_country_idx)
            region  = COUNTRY_REGION.get(country, "不明")
            st.text_input("地域（自動）", value=region, disabled=True)
            sector  = st.selectbox("セクター", sectors, index=default_sector_idx)
            subsectors = SUBSECTORS.get(sector, [])
            subsector  = st.selectbox("サブセクター", subsectors)

        with col2:
            scheme   = st.selectbox("協力スキーム", schemes, index=default_scheme_idx)
            budget   = st.number_input(
                "事業費（億円）",
                min_value=float(budget_cfg.get("min", 0.1)),
                max_value=float(budget_cfg.get("max", 500.0)),
                value=float(budget_cfg.get("default", 8.5)),
                step=float(budget_cfg.get("step", 0.1)),
                format="%.1f",
            )
            duration = st.slider(
                "事業期間（ヶ月）",
                min_value=int(duration_cfg.get("min", 6)),
                max_value=int(duration_cfg.get("max", 120)),
                value=int(duration_cfg.get("default", 36)),
                step=int(duration_cfg.get("step", 1)),
            )
            years, months = divmod(duration, 12)
            hint = f"（{years}年{months}ヶ月）" if years and months else f"（{years}年）" if years else ""
            st.caption(f"{duration} ヶ月 {hint}")

        overview = st.text_area("事業概要（任意）", placeholder="事業の目的・内容を簡潔に記載...", height=80)

        submitted = st.form_submit_button("▶ スクリーニングを実行", use_container_width=True)

    if submitted:
        st.session_state.input_data = {
            "country": country, "region": region, "sector": sector,
            "subsector": subsector, "scheme": scheme,
            "budget": budget, "duration": duration, "overview": overview,
        }
        with st.spinner("過去の類似案件を分析中..."):
            d = st.session_state.input_data
            st.session_state.risk_score   = calculate_mock_risk(
                d["country"], d["region"], d["sector"], d["scheme"], d["budget"], d["duration"]
            )
            st.session_state.shap_factors = generate_mock_shap(
                d["country"], d["region"], d["sector"], d["scheme"], d["budget"], d["duration"]
            )
        st.session_state.step = 2
        st.rerun()

# ──────────────────────────────────────────────────────────────
# STEP 2: リスク予測結果
# ──────────────────────────────────────────────────────────────
elif st.session_state.step == 2:
    d           = st.session_state.input_data
    risk_score  = st.session_state.risk_score
    shap_factors = st.session_state.shap_factors

    # 入力サマリーバッジ
    badges = " ".join(
        f'<span style="background:#f1f5f9;border:1px solid #e2e8f0;border-radius:20px;'
        f'padding:4px 12px;font-size:0.82rem;color:#475569;font-weight:500">{b}</span>'
        for b in [
            f"🌍 {d['country']}",
            f"🏗️ {d['sector']}（{d['subsector']}）",
            f"📋 {d['scheme']}",
            f"💰 {d['budget']}億円",
            f"⏱️ {d['duration']}ヶ月",
        ]
    )
    st.markdown(f'<div style="margin-bottom:20px">{badges}</div>', unsafe_allow_html=True)

    # ゲージ & 判定
    col_gauge, col_right = st.columns([1, 1])
    with col_gauge:
        st.markdown('<div class="section-header">📊 リスク確率</div>', unsafe_allow_html=True)
        st.plotly_chart(_build_gauge(risk_score), use_container_width=True, key="gauge")
        if risk_score < 40:
            st.markdown('<div class="verdict-good">✅ 良好な評価が見込まれます</div>', unsafe_allow_html=True)
        elif risk_score < 60:
            st.markdown('<div class="verdict-warning">⚠️ 一部リスクが予想されます</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-danger">🔴 要注意：重点的なリスク対策を推奨</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">🎯 評価項目別予測</div>', unsafe_allow_html=True)
        ratings = generate_sub_ratings(risk_score)
        _render_rating_cards(ratings)

        st.markdown('<div class="section-header">📈 SHAP 要因分析</div>', unsafe_allow_html=True)
        st.plotly_chart(_build_shap_chart(shap_factors), use_container_width=True, key="shap")

    # SHAP 説明
    st.markdown('<div class="section-header">🔍 要因別の解説</div>', unsafe_allow_html=True)
    _render_shap_explanations(shap_factors)

    # 類似案件
    st.markdown('<div class="section-header">📁 類似案件の評価実績</div>', unsafe_allow_html=True)
    similar = select_similar_projects(d["region"], d["sector"])
    _render_similar_projects(similar)

    st.markdown("---")
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("⬅ 入力に戻る", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col_btn2:
        if st.button("📚 教訓・対策を確認 ▶", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ──────────────────────────────────────────────────────────────
# STEP 3: 教訓・対策
# ──────────────────────────────────────────────────────────────
elif st.session_state.step == 3:
    d = st.session_state.input_data

    st.markdown('<div class="section-header">📚 過去の評価書から抽出された教訓・対策</div>', unsafe_allow_html=True)
    st.caption(
        f"対象条件：{d['country']} / {d['sector']}（{d['subsector']}）/ {d['scheme']} / "
        f"{d['budget']}億円 / {d['duration']}ヶ月"
    )

    lesson_data = select_lesson_data(d["region"], d["sector"])
    _render_lesson_panels(lesson_data)
    _render_summary(lesson_data["summary"])

    st.markdown("---")
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("⬅ リスク予測に戻る", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col_btn2:
        if st.button("🔄 新規分析を開始", type="primary", use_container_width=True):
            st.session_state.step = 1
            st.session_state.input_data = {}
            st.session_state.risk_score = None
            st.session_state.shap_factors = []
            st.rerun()
