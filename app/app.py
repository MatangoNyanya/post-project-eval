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

CUSTOM_CSS = """
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #1a1d27;
    border-right: 1px solid #2d3148;
}
[data-testid="stSidebar"] * {
    color: #c8cfe0 !important;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d1b69 50%, #1a3a2e 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}

/* ── Section header ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #1e3a5f;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.metric-card {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    border-color: #60a5fa;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 3px;
}

/* ── Stats table ── */
.stats-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
    margin-bottom: 20px;
}
.stats-table th {
    background: #1e2130;
    color: #60a5fa;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #2d3148;
}
.stats-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #1e2130;
    color: #cbd5e1;
}
.stats-table tr:hover td {
    background: #1e2130;
}
.badge-high { color: #f87171; font-weight: 700; }
.badge-mid  { color: #fb923c; font-weight: 700; }
.badge-low  { color: #34d399; font-weight: 700; }

/* ── LLM result card ── */
.llm-card-top {
    display: none;
}
[data-testid="stVerticalBlock"]:has(.llm-card-top) {
    background: linear-gradient(135deg, #1a2744 0%, #1a1f35 100%);
    border: 1px solid #2d3a5e;
    border-radius: 14px;
    padding: 20px 24px;
    margin-top: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
p.llm-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #a78bfa;
    margin: 0 0 16px 0;
}
p.llm-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #60a5fa;
    margin-bottom: 4px;
}
.llm-action-box {
    background: #0f2818;
    border-left: 4px solid #34d399;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.92rem;
    line-height: 1.75;
    color: #86efac;
}

/* ── Representative sentences ── */
.rep-card {
    background: #1a1d27;
    border: 1px solid #2d3148;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #cbd5e1;
    position: relative;
}
.rep-card:hover {
    border-color: #a78bfa;
    background: #1e2130;
}
.rep-rank {
    display: inline-block;
    background: #2d1b69;
    color: #a78bfa;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 8px;
}
.rep-meta {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 8px;
}

/* ── Info banner ── */
.info-banner {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    margin-top: 16px;
}
.info-banner .icon { font-size: 2.5rem; margin-bottom: 12px; }

/* ── Cluster selector ── */
div[data-testid="stSelectbox"] > label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* ── Form submit button ── */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 12px !important;
    width: 100% !important;
    transition: opacity 0.15s ease !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    opacity: 0.88 !important;
}

/* ── Sidebar form labels ── */
[data-testid="stSidebar"] label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2d3148; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #4a5080; }
</style>
"""


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
        paper_bgcolor="#0f1117",
        plot_bgcolor="#13161f",
        font=dict(color="#94a3b8", size=12),
        xaxis=dict(
            gridcolor="#1e2130",
            zerolinecolor="#2d3148",
            tickfont=dict(color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="#1e2130",
            zerolinecolor="#2d3148",
            tickfont=dict(color="#64748b"),
        ),
        hoverlabel=dict(
            bgcolor="#1e2130",
            bordercolor="#2d3148",
            font_color="#e2e8f0",
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
if LOGO_PATH.exists():
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image(str(LOGO_PATH), width=120)
    with col_title:
        st.markdown(
            """<div class="hero-header" style="margin-top:0">
                <div class="hero-title">LESMAP</div>
                <div class="hero-subtitle">条件を指定してクラスタリングを実行し、代表文と LLM による意味付けを確認できます。</div>
            </div>""",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """<div class="hero-header">
            <div class="hero-title">LESMAP</div>
            <div class="hero-subtitle">条件を指定してクラスタリングを実行し、代表文と LLM による意味付けを確認できます。</div>
        </div>""",
        unsafe_allow_html=True,
    )

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
