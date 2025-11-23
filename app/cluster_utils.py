from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap


def wrap_text(text: str, width: int = 40, html_break: bool = True) -> str:
    """Simple fixed-width wrapper that also works for Japanese text."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if not text:
        return ""
    chunks = [text[i : i + width] for i in range(0, len(text), width)]
    sep = "<br>" if html_break else "\n"
    return sep.join(chunks)


def _is_bad_score(value) -> float:
    """Return 1.0 if the evaluation is 1 or 2, else 0.0 (NaN preserved)."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(val):
        return np.nan
    return 1.0 if val in (1.0, 2.0) else 0.0


def _filter_dataframe(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    sectors: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    eval_year_range: Optional[Tuple[int, int]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    mask = pd.Series(True, index=df.index)
    if sectors:
        mask &= df["分野"].isin(list(sectors))
    if regions:
        mask &= df["region_detail"].isin(list(regions))
    if eval_year_range:
        years = pd.to_numeric(df.get("eval_year"), errors="coerce")
        start, end = eval_year_range
        mask &= years.between(start, end)

    filtered_df = df.loc[mask].reset_index(drop=True)
    filtered_embeddings = embeddings[mask.to_numpy()]
    return filtered_df, filtered_embeddings


def _compute_cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    def _agg(group: pd.DataFrame) -> pd.Series:
        bad_ratio = group["bad_flag"].mean()
        mask = group["bad_flag"].notna()
        if mask.any():
            weights = group.loc[mask, "doc_weight"]
            weighted = np.average(group.loc[mask, "bad_flag"], weights=weights)
        else:
            weighted = np.nan

        return pd.Series(
            {
                "n": len(group),
                "bad_ratio": bad_ratio,
                "bad_ratio_weighted": weighted,
            }
        )

    stats = (
        df.groupby("cluster")
        .apply(_agg)
        .reset_index()
        .sort_values("bad_ratio_weighted", ascending=False)
    )
    return stats


def _compute_umap_layout(
    vectors: np.ndarray, random_state: int = 42, min_dist: float = 0.1
) -> np.ndarray:
    n_samples = vectors.shape[0]
    if n_samples < 2:
        raise ValueError("UMAP requires at least two samples.")
    n_neighbors = min(15, max(2, n_samples - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


@dataclass
class ClusterConfig:
    n_clusters: int
    n_representatives: int
    pca_dim: int = 50
    random_state: int = 42


@dataclass
class ClusterResult:
    data: pd.DataFrame
    stats: pd.DataFrame
    representatives: pd.DataFrame
    filters: dict
    config: ClusterConfig


def run_clustering(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    config: ClusterConfig,
    sectors: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    eval_year_range: Optional[Tuple[int, int]] = None,
) -> ClusterResult:
    """Replicate the clustering.ipynb workflow in pure Python."""
    filtered_df, filtered_embeddings = _filter_dataframe(
        df,
        embeddings,
        sectors=sectors,
        regions=regions,
        eval_year_range=eval_year_range,
    )
    n_samples = len(filtered_df)

    if n_samples == 0:
        raise ValueError("選択された条件に一致するデータがありません。")
    if n_samples < 2:
        raise ValueError("クラスタリングには最低2レコードが必要です。")
    if config.n_clusters > n_samples:
        raise ValueError("クラスタ数 k がデータ件数を超えています。")

    filtered_df = filtered_df.copy()

    n_components = min(config.pca_dim, filtered_embeddings.shape[1], n_samples)
    if n_components < 2:
        n_components = min(2, filtered_embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=config.random_state)
    reduced = pca.fit_transform(filtered_embeddings)

    kmeans = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state,
        n_init=10,
    )
    filtered_df["cluster"] = kmeans.fit_predict(reduced)

    filtered_df["bad_flag"] = filtered_df["total_eval"].apply(_is_bad_score)

    doc_counts = filtered_df.groupby("project_id")["para_id"].size().to_dict()
    filtered_df["doc_weight"] = filtered_df["project_id"].map(
        lambda pid: 1.0 / doc_counts.get(pid, 1)
    )

    stats = _compute_cluster_stats(filtered_df)

    layout = _compute_umap_layout(reduced, random_state=config.random_state)
    filtered_df["umap_x"] = layout[:, 0]
    filtered_df["umap_y"] = layout[:, 1]

    centers = kmeans.cluster_centers_
    dists = np.linalg.norm(reduced - centers[filtered_df["cluster"].values], axis=1)
    filtered_df["cluster_dist"] = dists
    filtered_df["cluster_rank"] = (
        filtered_df.groupby("cluster")["cluster_dist"].rank(method="first")
    )

    representatives = (
        filtered_df.sort_values(["cluster", "cluster_dist"])
        .groupby("cluster")
        .head(config.n_representatives)
        .reset_index(drop=True)
    )
    representatives["hover_text"] = representatives["text"].apply(wrap_text)

    filters = {
        "sectors": list(sectors or []),
        "regions": list(regions or []),
        "eval_year_range": eval_year_range if eval_year_range else (),
    }

    return ClusterResult(
        data=filtered_df,
        stats=stats,
        representatives=representatives,
        filters=filters,
        config=config,
    )
