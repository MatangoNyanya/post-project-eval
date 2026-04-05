from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:  # optional dependency for click events
    from streamlit_plotly_events import plotly_events
except ImportError:  # pragma: no cover - optional dependency
    plotly_events = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cluster_utils import ClusterConfig, ClusterResult, run_clustering, wrap_text
from llm_utils import label_cluster

DATASET_CONFIGS = {
    "一文": {
        "data_path": ROOT_DIR
        / "learning"
        / "output"
        / "kadai_text_with_rating_and_clusters_kyokun_mini.csv",
        "embedding_path": ROOT_DIR / "learning" / "output" / "embeddings_mini.npy",
    },
    "400字": {
        "data_path": ROOT_DIR / "learning" / "output" / "kadai_text_with_rating_and_clusters_kyokun.csv",
        "embedding_path": ROOT_DIR / "learning" / "output" / "embeddings.npy",
    },
}
DEFAULT_DATASET_KEY = "400字"
CONFIG_PATH = ROOT_DIR / "app" / "config.yaml"
LOGO_PATH = ROOT_DIR / "app" / "logo.png"

DEFAULT_CONFIG = {
    "llm": {
        "label_sentence_count": 5,
    },
    "ui": {
        "cluster_slider": {"min": 2, "default": 20, "max_cap": 50, "step": 1},
        "representative_slider": {"min": 3, "max": 20, "default": 5, "step": 1},
    },
    "plot": {
        "hover_wrap_width": 40,
        "height": 650,
        "margin": {"l": 10, "r": 10, "t": 40, "b": 40},
        "marker": {
            "size": 8,
            "colorscale": "Turbo",
            "showscale": True,
            "colorbar": {"title": "cluster"},
            "opacity": 0.8,
        },
    },
}

CUSTOM_CSS = '''
<style>
/* ── Base Modern ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: #f8fafc;
    color: #0f172a;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * {
    color: #334155 !important;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 32px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #2563eb, #7c3aed, #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    margin: 0;
    font-weight: 400;
}
.hero-desc {
    font-size: 0.95rem;
    color: #64748b;
    line-height: 1.75;
    margin: -16px 0 24px 0;
    padding: 0 4px;
}

/* ── Section header ── */
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e293b;
    margin: 36px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.01em;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.04);
    border-color: #cbd5e1;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* ── Stats table ── */
.stats-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin-bottom: 24px;
    background: #ffffff;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);
}
.stats-table th {
    background: #f8fafc;
    color: #475569;
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid #e2e8f0;
}
.stats-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #f1f5f9;
    color: #334155;
}
.stats-table tr:hover td {
    background: #f8fafc;
}

/* ── LLM result card ── */
.llm-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 20px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.llm-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #7c3aed;
    margin: 0 0 20px 0;
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: -0.01em;
}
.llm-section-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 8px;
}
.llm-body {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #334155;
    margin-bottom: 20px;
}
.llm-action-box {
    background: #ecfdf5;
    border-left: 4px solid #10b981;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #065f46;
}

/* ── Representative sentences ── */
.rep-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #334155;
    position: relative;
    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.03);
    transition: all 0.2s ease;
}
.rep-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.rep-rank {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    margin-bottom: 12px;
}
.rep-meta {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 12px;
    font-weight: 500;
}

/* ── Info banner ── */
.info-banner {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 48px;
    text-align: center;
    color: #1e3a8a;
    font-size: 1.1rem;
    margin-top: 24px;
    font-weight: 500;
}
.info-banner .icon { font-size: 3rem; margin-bottom: 16px; }

/* ── Cluster selector ── */
div[data-testid="stSelectbox"] > label {
    color: #475569 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* ── Form submit button ── */
div[data-testid="stFormSubmitButton"] > button {
    background: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #1e293b !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 8px -1px rgba(0,0,0,0.15) !important;
}
div[data-testid="stFormSubmitButton"] > button p,
div[data-testid="stFormSubmitButton"] > button span {
    color: #ffffff !important;
}

/* ── Sidebar form labels ── */
[data-testid="stSidebar"] label {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #334155;
    background: #f8fafc;
    border-radius: 8px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
'''


def _deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_app_config(path: Path = CONFIG_PATH) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if not path.exists() or yaml is None:
        if yaml is None:
            st.warning("pyyaml が見つかりません。`pip install pyyaml` でインストールしてください。")
        return config
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem errors
        st.warning(f"設定ファイルの読み込みに失敗しました: {exc}")
        return config
    try:
        loaded = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - YAML syntax errors
        st.warning(f"設定ファイルの解析に失敗しました: {exc}")
        return config
    return _deep_update(config, loaded)


APP_CONFIG = _load_app_config()
UI_CONFIG = APP_CONFIG.get("ui", {})
LLM_CONFIG = APP_CONFIG.get("llm", {})
PLOT_CONFIG = APP_CONFIG.get("plot", {})

st.set_page_config(
    page_title="LESMAP | クラスタ分析ダッシュボード",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource(show_spinner=False)
def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def _build_scatter_figure(df: pd.DataFrame, plot_cfg: dict) -> go.Figure:
    wrap_width = plot_cfg.get("hover_wrap_width", 40)
    hover_text = df["text"].apply(lambda x: wrap_text(x, width=wrap_width))
    customdata = np.stack((df["cluster"].values,), axis=-1)
    marker = deepcopy(plot_cfg.get("marker", {}))
    marker["color"] = df["cluster"]
    marker.setdefault("colorbar", {"title": "cluster"})
    scatter = go.Scattergl(
        x=df["umap_x"],
        y=df["umap_y"],
        mode="markers",
        marker=marker,
        text=hover_text,
        customdata=customdata,
        hovertemplate="<b>クラスタ %{customdata[0]}</b><br>%{text}<extra></extra>",
    )
    fig = go.Figure(data=[scatter])
    fig.update_layout(
        height=plot_cfg.get("height", 650),
        margin=plot_cfg.get("margin", dict(l=10, r=10, t=40, b=40)),
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#475569", size=12, family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            tickfont=dict(color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            tickfont=dict(color="#64748b"),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#e2e8f0",
            font_color="#0f172a",
        ),
    )
    return fig


def _parse_llm_output(text: str) -> dict[str, str]:
    """Parse 'タイトル: ...\n説明: ...\n対策: ...' format."""
    result = {"title": "", "description": "", "action": "", "raw": text}
    title_m = re.search(r"タイトル[：:]\s*(.+?)(?:\n|$)", text)
    desc_m = re.search(r"説明[：:]\s*([\s\S]+?)(?:\n対策[：:]|$)", text)
    act_m = re.search(r"対策[：:]\s*([\s\S]+?)$", text.strip())
    if title_m:
        result["title"] = title_m.group(1).strip()
    if desc_m:
        result["description"] = desc_m.group(1).strip()
    if act_m:
        result["action"] = act_m.group(1).strip()
    return result


def _render_llm_card(label_text: str) -> None:
    parsed = _parse_llm_output(label_text)
    with st.container():
        st.markdown('<div class="llm-card-top"></div>', unsafe_allow_html=True)
        if parsed["title"]:
            st.markdown(f'<p class="llm-title">💡 {parsed["title"]}</p>', unsafe_allow_html=True)
        if parsed["description"]:
            st.markdown('<p class="llm-section-label">📋 説明</p>', unsafe_allow_html=True)
            st.markdown(parsed["description"])
        if parsed["action"]:
            st.markdown('<p class="llm-section-label">🔧 対策</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="llm-action-box">{parsed["action"]}</div>',
                unsafe_allow_html=True,
            )
        if not parsed["title"] and not parsed["description"] and not parsed["action"]:
            st.markdown(label_text)


def _render_stats_section(stats: pd.DataFrame) -> None:
    st.markdown('<div class="section-header">📊 クラスタ統計</div>', unsafe_allow_html=True)

    total_docs = int(stats["n"].sum()) if "n" in stats.columns else 0
    avg_bad = stats["bad_ratio_weighted"].mean() if "bad_ratio_weighted" in stats.columns else float("nan")
    top_bad_cluster = (
        int(stats.iloc[0]["cluster"]) if not stats.empty and "cluster" in stats.columns else "—"
    )
    top_bad_ratio = (
        float(stats.iloc[0]["bad_ratio_weighted"]) if not stats.empty else float("nan")
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("総データ数", f"{total_docs:,}")
    with c2:
        st.metric("クラスタ数", len(stats))
    with c3:
        st.metric("平均不良率 (加重)", f"{avg_bad:.1%}" if not pd.isna(avg_bad) else "—")
    with c4:
        st.metric(
            "最も不良率が高いクラスタ",
            f"#{top_bad_cluster}",
            delta=f"{top_bad_ratio:.1%}" if not pd.isna(top_bad_ratio) else None,
            delta_color="inverse",
        )

    with st.expander("全クラスタ統計テーブルを表示", expanded=False):
        display_stats = stats.copy()
        if "bad_ratio" in display_stats.columns:
            display_stats["bad_ratio"] = display_stats["bad_ratio"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
        if "bad_ratio_weighted" in display_stats.columns:
            display_stats["bad_ratio_weighted"] = display_stats["bad_ratio_weighted"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"
            )
        display_stats = display_stats.rename(columns={
            "cluster": "クラスタ",
            "n": "分類数",
            "bad_ratio": "課題ありプロジェクト率",
            "bad_ratio_weighted": "課題ありプロジェクト率（加重）",
        })
        st.dataframe(
            display_stats,
            use_container_width=True,
            hide_index=True,
            column_config={
                "クラスタ": st.column_config.TextColumn(
                    "クラスタ",
                    help="クラスタリングで割り当てられたクラスタID",
                ),
                "分類数": st.column_config.TextColumn(
                    "分類数",
                    help="このクラスタに分類されたテキスト（段落）の件数",
                ),
                "課題ありプロジェクト率": st.column_config.TextColumn(
                    "課題ありプロジェクト率",
                    help="評価スコアが1または2（低評価）だったプロジェクトの割合。各テキストを1件1票として単純に集計。",
                ),
                "課題ありプロジェクト率（加重）": st.column_config.TextColumn(
                    "課題ありプロジェクト率（加重）",
                    help="課題ありプロジェクト率を、プロジェクトごとのテキスト件数で逆数重み付けして算出。テキスト数が多いプロジェクトの影響を補正した指標。",
                ),
            },
        )


def _render_representatives(result: ClusterResult, cluster_id: int, llm_cfg: dict) -> None:
    reps = result.representatives[result.representatives["cluster"] == cluster_id]
    if reps.empty:
        st.markdown(
            '<div class="info-banner"><div class="icon">📭</div>このクラスタには代表文がありません。</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="section-header">📝 クラスタ {cluster_id} の代表文</div>',
        unsafe_allow_html=True,
    )

    reps_sorted = reps.sort_values("cluster_dist").reset_index(drop=True)
    for i, row in reps_sorted.iterrows():
        rank = int(row.get("cluster_rank", i + 1))
        dist = float(row.get("cluster_dist", 0))
        pid = row.get("project_id", "—")
        para = row.get("para_id", "—")
        text = str(row.get("text", ""))
        st.markdown(
            f"""<div class="rep-card">
                <span class="rep-rank">#{rank} 代表文</span>
                <div>{text}</div>
                <div class="rep-meta">project: {pid} &nbsp;|&nbsp; para: {para} &nbsp;|&nbsp; dist: {dist:.4f}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    sentence_limit = int(llm_cfg.get("label_sentence_count", 5))
    top_sentences = reps_sorted["text"].head(sentence_limit).tolist()

    st.markdown('<div class="section-header">🤖 LLM によるクラスタ意味付け</div>', unsafe_allow_html=True)
    with st.spinner("LLM で分析中..."):
        label_text = label_cluster(tuple(top_sentences))
    _render_llm_card(label_text)


# ── Header ──────────────────────────────────────────────────────────────────
_desc1 = "事後評価の「教訓」に記載された言葉をクラスタリングで分類するツールです。"
_desc2 = "分野・地域・評価年などの条件を指定して実行すると、各クラスタの代表文をもとに LLM が共通テーマ・対策を推論します。"

_hero_inner = (
    '<div class="hero-title">LESMAP</div>'
    f'<div class="hero-subtitle">{_desc1}<br>{_desc2}</div>'
)

if LOGO_PATH.exists():
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image(str(LOGO_PATH), width=120)
    with col_title:
        st.markdown(f'<div class="hero-header" style="margin-top:0">{_hero_inner}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="hero-header">{_hero_inner}</div>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
dset_keys = list(DATASET_CONFIGS.keys())
default_dataset_index = (
    dset_keys.index(DEFAULT_DATASET_KEY) if DEFAULT_DATASET_KEY in dset_keys else 0
)

with st.sidebar:
    st.markdown("### ⚙️ 設定")
    st.markdown("---")

    selected_dataset_key = st.radio(
        "文章タイプ",
        options=dset_keys,
        index=default_dataset_index,
        key="text_length_radio",
    )

    st.markdown("---")
    st.markdown("### 🔍 クラスタリング条件")

    selected_dataset = DATASET_CONFIGS[selected_dataset_key]
    selected_data_path = selected_dataset["data_path"]
    selected_embedding_path = selected_dataset["embedding_path"]

    if not selected_data_path.exists():
        st.error(f"データファイルが見つかりません: {selected_data_path}")
        st.stop()
    if not selected_embedding_path.exists():
        st.error(f"埋め込みファイルが見つかりません: {selected_embedding_path}")
        st.stop()

    df_master = load_dataframe(str(selected_data_path))
    embeddings_master = load_embeddings(str(selected_embedding_path))

    form = st.form("controls")

    year_series = pd.to_numeric(df_master.get("eval_year"), errors="coerce")
    valid_years = year_series.dropna()
    if valid_years.empty:
        eval_year_min = eval_year_max = 0
    else:
        eval_year_min = int(valid_years.min())
        eval_year_max = int(valid_years.max())

    selected_eval_years = form.slider(
        "評価年",
        min_value=eval_year_min,
        max_value=eval_year_max,
        value=(eval_year_min, eval_year_max),
        step=1,
    )

    cluster_slider_cfg = UI_CONFIG.get("cluster_slider", {})
    cluster_min = cluster_slider_cfg.get("min", 2)
    cluster_step = cluster_slider_cfg.get("step", 1)
    cluster_max_cap = cluster_slider_cfg.get("max_cap", 50)
    cluster_default = cluster_slider_cfg.get("default", 20)
    max_k = max(cluster_min, min(cluster_max_cap, len(df_master)))
    k_default = min(max_k, max(cluster_min, cluster_default))
    k = form.slider(
        "クラスタ数 (k)",
        min_value=cluster_min,
        max_value=max_k,
        value=k_default,
        step=cluster_step,
    )

    representative_cfg = UI_CONFIG.get("representative_slider", {})
    rep_min = representative_cfg.get("min", 3)
    rep_max = representative_cfg.get("max", 20)
    rep_step = representative_cfg.get("step", 1)
    rep_default = representative_cfg.get("default", 5)
    rep_default = min(rep_max, max(rep_min, rep_default))
    rep_num = form.slider(
        "代表文の件数 (N)",
        min_value=rep_min,
        max_value=rep_max,
        value=rep_default,
        step=rep_step,
    )

    sector_options = sorted(df_master["分野"].dropna().unique().tolist())
    region_options = sorted(df_master["region_detail"].dropna().unique().tolist())

    selected_sectors = form.multiselect("分野", sector_options)
    selected_regions = form.multiselect("地域 (region_detail)", region_options)

    submitted = form.form_submit_button("▶ クラスタリングを実行", use_container_width=True)

# ── Clustering execution ──────────────────────────────────────────────────────
if submitted:
    with st.spinner("クラスタリングを実行しています..."):
        try:
            cfg = ClusterConfig(n_clusters=k, n_representatives=rep_num)
            session_result = run_clustering(
                df_master,
                embeddings_master,
                cfg,
                sectors=selected_sectors,
                regions=selected_regions,
                eval_year_range=selected_eval_years,
            )
        except ValueError as exc:
            st.error(str(exc))
            session_result = None
        else:
            st.session_state["cluster_result"] = session_result

result: ClusterResult | None = st.session_state.get("cluster_result")

if result is None:
    st.markdown(
        """<div class="info-banner">
            <div class="icon">🚀</div>
            左のサイドバーから条件を設定し、「クラスタリングを実行」を押してください。
        </div>""",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
_render_stats_section(result.stats)

st.markdown('<div class="section-header">🗺️ UMAP × Plotly 可視化</div>', unsafe_allow_html=True)
figure = _build_scatter_figure(result.data, PLOT_CONFIG)
st.plotly_chart(figure, use_container_width=True, key="umap_plot")

available_clusters = sorted(result.representatives["cluster"].unique().tolist())
if not available_clusters:
    st.markdown(
        '<div class="info-banner"><div class="icon">📭</div>代表文が存在するクラスタがありません。</div>',
        unsafe_allow_html=True,
    )
else:
    default_cluster = st.session_state.get("selected_cluster", available_clusters[0])
    if default_cluster not in available_clusters:
        default_cluster = available_clusters[0]
    default_index = available_clusters.index(default_cluster)

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        selected_cluster = st.selectbox(
            "代表文を確認するクラスタを選択",
            options=available_clusters,
            index=default_index,
            key="cluster_selectbox",
        )
    with col_info:
        if "bad_ratio_weighted" in result.stats.columns:
            row = result.stats[result.stats["cluster"] == selected_cluster]
            if not row.empty:
                br = float(row.iloc[0]["bad_ratio_weighted"])
                n = int(row.iloc[0]["n"])
                c_a, c_b = st.columns(2)
                c_a.metric("データ件数", f"{n:,}")
                c_b.metric("不良率 (加重)", f"{br:.1%}" if not pd.isna(br) else "—")

    st.session_state["selected_cluster"] = selected_cluster
    _render_representatives(result, selected_cluster, LLM_CONFIG)
