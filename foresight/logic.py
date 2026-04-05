# ==============================================
# Foresight - リスク予測ロジック
# ==============================================
from __future__ import annotations

import math

from data import (
    AFRICA_WATER_LESSONS,
    DEFAULT_LESSONS,
    SIMILAR_PROJECTS_DEFAULT,
    SIMILAR_PROJECTS_EDUCATION,
    SIMILAR_PROJECTS_WATER,
    SOUTHASIA_EDUCATION_LESSONS,
)


# ──────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ──────────────────────────────────────────────────────────────

def _hash_code(s: str) -> int:
    h = 0
    for ch in s:
        h = ((h << 5) - h) + ord(ch)
        h &= 0xFFFFFFFF
        if h >= 0x80000000:
            h -= 0x100000000
    return abs(h)


def _seeded_random(seed: int) -> float:
    x = math.sin(seed) * 10000
    return x - math.floor(x)


def _round2(n: float) -> float:
    return round(n * 100) / 100


# ──────────────────────────────────────────────────────────────
# 説明文ヘルパー
# ──────────────────────────────────────────────────────────────

_COUNTRY_EXPLANATION: dict[str, str] = {
    "サブサハラアフリカ": "サブサハラアフリカ地域は制度・財務基盤が相対的に脆弱。持続性の課題が頻出",
    "南アジア":           "南アジア地域。行政能力にばらつきがあり、事業管理体制の構築が課題になりやすい",
    "東南アジア":         "東南アジア地域。CLMV諸国では制度基盤の発展途上にあり、先発ASEAN諸国と差がある",
    "中南米":             "中南米地域。制度基盤は比較的整備されているが、政権交代による政策変更リスクあり",
    "中東・北アフリカ":   "中東・北アフリカ地域。治安・政情の変動リスクが事業実施に影響する可能性",
}

_SECTOR_EXPLANATION: dict[str, str] = {
    "運輸交通":     "インフラ完成後の維持管理に経常的コストが発生。持続性の課題が出やすいセクター",
    "水資源・防災": "施設の維持管理コストと料金徴収体制の構築が持続性の鍵",
    "保健医療":     "ソフト面中心の支援が多く、比較的安定した評価が得られる傾向",
    "教育":         "人材育成効果が長期的に波及しやすく、評価が安定する傾向",
    "農業・農村開発": "自然環境・市場条件の変動影響を受けやすい",
    "ガバナンス":   "制度改革の成果が可視化しにくく、有効性の評価が難しい傾向",
    "資源・エネルギー": "大規模インフラが多く、コスト超過・工期遅延のリスクあり",
    "都市開発・環境": "複数ステークホルダーの調整が必要で、効率性に課題が出やすい",
    "情報通信技術": "技術革新が速く、事業完了時に陳腐化するリスク",
}

_SCHEME_EXPLANATION: dict[str, str] = {
    "技術協力":     "ソフト面の支援が中心。管理しやすく、評価が比較的安定する傾向",
    "有償資金協力": "大規模インフラが多い。コスト超過・工期遅延のリスクが相対的に高い",
    "無償資金協力": "施設建設が中心。維持管理の引き渡し後の持続性が課題になりやすい",
}


# ──────────────────────────────────────────────────────────────
# 公開関数
# ──────────────────────────────────────────────────────────────

def calculate_mock_risk(
    country: str,
    region: str,
    sector: str,
    scheme: str,
    budget: float,
    duration: int,
) -> float:
    """ダミーのリスク確率（%）を計算する。"""
    base_risk = 35.0

    region_risk = {
        "サブサハラアフリカ": 20,
        "南アジア":           12,
        "東南アジア":          8,
        "中南米":              5,
        "中東・北アフリカ":   10,
    }
    base_risk += region_risk.get(region, 5)

    sector_risk = {
        "運輸交通":       8,
        "水資源・防災":   6,
        "資源・エネルギー": 5,
        "都市開発・環境": 4,
        "農業・農村開発": 3,
        "ガバナンス":     7,
        "保健医療":      -2,
        "教育":          -3,
        "情報通信技術":   2,
    }
    base_risk += sector_risk.get(sector, 0)

    scheme_risk = {"有償資金協力": 8, "無償資金協力": 3, "技術協力": 0}
    base_risk += scheme_risk.get(scheme, 0)

    if budget > 100:
        base_risk += 10
    elif budget > 50:
        base_risk += 5
    elif budget < 1:
        base_risk += 3

    if duration > 72:
        base_risk += 8
    elif duration > 48:
        base_risk += 4

    seed = _hash_code(country + sector + str(budget))
    noise = _seeded_random(seed) * 6 - 3

    return max(5.0, min(95.0, base_risk + noise))


def generate_mock_shap(
    country: str,
    region: str,
    sector: str,
    scheme: str,
    budget: float,
    duration: int,
) -> list[dict]:
    """ダミーのSHAP値リストを生成する。"""
    seed = _hash_code(country + sector + scheme)
    factors: list[dict] = []

    # 国/地域
    country_shap_base = {
        "サブサハラアフリカ": 0.22,
        "南アジア":           0.15,
        "東南アジア":         0.10,
        "中南米":             0.05,
        "中東・北アフリカ":   0.12,
    }
    country_val = _round2((country_shap_base.get(region, 0.08)) + _seeded_random(seed) * 0.06 - 0.03)
    factors.append({
        "factor":      f"国: {country}",
        "value":       country_val,
        "icon":        "🌍",
        "direction":   "risk" if country_val > 0 else "safe",
        "explanation": _COUNTRY_EXPLANATION.get(region, "対象国の制度・実施環境が評価に影響する傾向"),
    })

    # セクター
    sector_shap_base = {
        "運輸交通": 0.11, "水資源・防災": 0.08, "ガバナンス": 0.09,
        "資源・エネルギー": 0.07, "農業・農村開発": 0.04,
        "都市開発・環境": 0.05, "保健医療": -0.03, "教育": -0.04, "情報通信技術": 0.02,
    }
    sector_val = _round2((sector_shap_base.get(sector, 0.03)) + _seeded_random(seed + 1) * 0.04 - 0.02)
    factors.append({
        "factor":      f"セクター: {sector}",
        "value":       sector_val,
        "icon":        "🏗️",
        "direction":   "risk" if sector_val > 0 else "safe",
        "explanation": _SECTOR_EXPLANATION.get(sector, ""),
    })

    # スキーム
    scheme_shap = {"有償資金協力": 0.09, "無償資金協力": 0.03, "技術協力": -0.03}
    scheme_val = _round2(scheme_shap.get(scheme, 0.0))
    factors.append({
        "factor":      f"スキーム: {scheme}",
        "value":       scheme_val,
        "icon":        "📋",
        "direction":   "risk" if scheme_val > 0 else "safe",
        "explanation": _SCHEME_EXPLANATION.get(scheme, ""),
    })

    # 事業費
    if budget > 100:
        budget_val = _round2(0.12 + _seeded_random(seed + 2) * 0.02 - 0.01)
        budget_exp = "大規模案件。管理コストとステークホルダー調整の複雑性が増大"
    elif budget > 50:
        budget_val = _round2(0.06 + _seeded_random(seed + 2) * 0.02 - 0.01)
        budget_exp = "中〜大規模。効率性管理に注意が必要"
    elif budget > 10:
        budget_val = _round2(-0.02 + _seeded_random(seed + 2) * 0.02 - 0.01)
        budget_exp = "中規模案件。管理が比較的容易で効率性が維持されやすい"
    else:
        budget_val = _round2(-0.05 + _seeded_random(seed + 2) * 0.02 - 0.01)
        budget_exp = "小〜中規模案件。管理コストが抑えられる傾向"
    factors.append({
        "factor":      f"事業費: {budget}億円",
        "value":       budget_val,
        "icon":        "💰",
        "direction":   "risk" if budget_val > 0 else "safe",
        "explanation": budget_exp,
    })

    # 事業期間
    if duration > 72:
        dur_val = _round2(0.10 + _seeded_random(seed + 3) * 0.02 - 0.01)
        dur_exp = "長期事業。カウンターパート異動や外部環境変化のリスクが蓄積"
    elif duration > 48:
        dur_val = _round2(0.05 + _seeded_random(seed + 3) * 0.02 - 0.01)
        dur_exp = "やや長期。C/P異動リスクが累積する傾向"
    elif duration > 24:
        dur_val = _round2(-0.02 + _seeded_random(seed + 3) * 0.02 - 0.01)
        dur_exp = "標準的な事業期間。大きなリスク要因ではない"
    else:
        dur_val = _round2(-0.04 + _seeded_random(seed + 3) * 0.02 - 0.01)
        dur_exp = "短期事業。迅速な実施が見込める"
    factors.append({
        "factor":      f"事業期間: {duration}ヶ月",
        "value":       dur_val,
        "icon":        "⏱️",
        "direction":   "risk" if dur_val > 0 else "safe",
        "explanation": dur_exp,
    })

    # 絶対値でソート（大きい順）
    factors.sort(key=lambda f: abs(f["value"]), reverse=True)
    return factors


def select_lesson_data(region: str, sector: str) -> dict:
    """入力条件に応じて教訓データを選択する。"""
    if "アフリカ" in region and sector == "水資源・防災":
        return AFRICA_WATER_LESSONS
    if region == "南アジア" and sector == "教育":
        return SOUTHASIA_EDUCATION_LESSONS
    return DEFAULT_LESSONS


def select_similar_projects(region: str, sector: str) -> list[dict]:
    """入力条件に応じて類似案件データを選択する。"""
    if "アフリカ" in region and sector == "水資源・防災":
        return SIMILAR_PROJECTS_WATER
    if region == "南アジア" and sector == "教育":
        return SIMILAR_PROJECTS_EDUCATION
    return SIMILAR_PROJECTS_DEFAULT


def generate_sub_ratings(risk_prob: float) -> list[dict]:
    """評価項目別のダミー予測レーティングを返す。"""
    def _get_rating(prob: float, bias: float) -> dict:
        adjusted = prob + bias
        if adjusted < 40:
            return {"rating": "③", "label": "良好",   "cls": "rating-3"}
        if adjusted < 65:
            return {"rating": "②", "label": "やや懸念", "cls": "rating-2"}
        return {"rating": "①", "label": "要注意", "cls": "rating-1"}

    return [
        {"name": "妥当性", **_get_rating(risk_prob, -25)},
        {"name": "有効性", **_get_rating(risk_prob,  -5)},
        {"name": "効率性", **_get_rating(risk_prob,   5)},
        {"name": "持続性", **_get_rating(risk_prob,  12)},
    ]
