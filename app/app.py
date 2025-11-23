from __future__ import annotations

from copy import deepcopy
from pathlib import Path
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

from app.cluster_utils import ClusterConfig, ClusterResult, run_clustering, wrap_text
from app.llm_utils import label_cluster

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
            "size": 10,
            "colorscale": "Rainbow",
            "showscale": True,
            "colorbar": {"title": "cluster"},
            "opacity": 0.85,
        },
    },
}


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

st.set_page_config(page_title="クラスタ分析ダッシュボード", layout="wide")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=350)
st.title("LESMAP")
st.caption("条件を指定してクラスタリングを実行し、代表文とLLMによる意味付けを確認できます。")


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
        hovertemplate="クラスタ %{customdata[0]}<br>%{text}<extra></extra>",
    )
    fig = go.Figure(data=[scatter])
    fig.update_layout(
        height=plot_cfg.get("height", 650),
        margin=plot_cfg.get("margin", dict(l=10, r=10, t=40, b=40)),
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
    )
    return fig


def _render_representatives(result: ClusterResult, cluster_id: int, llm_cfg: dict) -> None:
    reps = result.representatives[result.representatives["cluster"] == cluster_id]
    if reps.empty:
        st.info("このクラスタには代表文がありません。")
        return

    st.subheader(f"クラスタ {cluster_id} の代表文")
    display_cols = ["project_id", "para_id", "cluster_dist", "text"]
    st.dataframe(reps[display_cols], use_container_width=True)

    sentence_limit = int(llm_cfg.get("label_sentence_count", 5))
    top_sentences = (
        reps.sort_values("cluster_dist")["text"].head(sentence_limit).tolist()
    )
    with st.spinner("LLMでクラスタ意味付け中..."):
        label_text = label_cluster(tuple(top_sentences))
    st.success(label_text)


dset_keys = list(DATASET_CONFIGS.keys())
default_dataset_index = (
    dset_keys.index(DEFAULT_DATASET_KEY) if DEFAULT_DATASET_KEY in dset_keys else 0
)
selected_dataset_key = st.sidebar.radio(
    "文章タイプ",
    options=dset_keys,
    index=default_dataset_index,
    key="text_length_radio",
)
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

with st.sidebar:
    st.header("クラスタリング条件")
    form = st.form("controls")
    year_series = pd.to_numeric(df_master.get("eval_year"), errors="coerce")
    valid_years = year_series.dropna()
    if valid_years.empty:
        eval_year_min = 0
        eval_year_max = 0
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
    selected_regions = form.multiselect("region_detail", region_options)

    submitted = form.form_submit_button("クラスタリングを実行")

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
    st.info("左のフォームから条件を設定し、「クラスタリングを実行」を押してください。")
    st.stop()

st.subheader("クラスタ統計")
st.dataframe(result.stats, use_container_width=True)

figure = _build_scatter_figure(result.data, PLOT_CONFIG)
st.subheader("UMAP × Plotly 可視化")

st.plotly_chart(figure, use_container_width=True, key="umap_plot")

available_clusters = sorted(result.representatives["cluster"].unique().tolist())
if not available_clusters:
    st.info("代表文が存在するクラスタがありません。")
else:
    default_cluster = st.session_state.get("selected_cluster", available_clusters[0])
    if default_cluster not in available_clusters:
        default_cluster = available_clusters[0]

    default_index = available_clusters.index(default_cluster)
    selected_cluster = st.selectbox(
        "代表文を確認するクラスタを選択",
        options=available_clusters,
        index=default_index,
        key="cluster_selectbox",
    )
    st.session_state["selected_cluster"] = selected_cluster
    _render_representatives(result, selected_cluster, LLM_CONFIG)
