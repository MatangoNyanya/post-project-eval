// ==============================================
// EvalNavi - ダミーロジック（予測・SHAP生成）
// ==============================================

/**
 * ダミーのリスク確率計算ロジック
 */
function calculateMockRisk(country, region, sector, scheme, budget, duration) {
  let baseRisk = 35;

  // 地域リスク
  const regionRisk = {
    "サブサハラアフリカ": 20,
    "南アジア": 12,
    "東南アジア": 8,
    "中南米": 5,
    "中東・北アフリカ": 10
  };
  baseRisk += (regionRisk[region] || 5);

  // セクターリスク
  const sectorRisk = {
    "運輸交通": 8,
    "水資源・防災": 6,
    "資源・エネルギー": 5,
    "都市開発・環境": 4,
    "農業・農村開発": 3,
    "ガバナンス": 7,
    "保健医療": -2,
    "教育": -3,
    "情報通信技術": 2
  };
  baseRisk += (sectorRisk[sector] || 0);

  // スキームリスク
  const schemeRisk = {
    "有償資金協力": 8,
    "無償資金協力": 3,
    "技術協力": 0
  };
  baseRisk += (schemeRisk[scheme] || 0);

  // 予算規模
  if (budget > 100) baseRisk += 10;
  else if (budget > 50) baseRisk += 5;
  else if (budget < 1) baseRisk += 3;

  // 事業期間
  if (duration > 72) baseRisk += 8;
  else if (duration > 48) baseRisk += 4;

  // ノイズ（シード固定）
  const seed = hashCode(country + sector + String(budget));
  const noise = seededRandom(seed) * 6 - 3; // -3 to +3

  return Math.max(5, Math.min(95, baseRisk + noise));
}

/**
 * ダミーのSHAP値生成
 */
function generateMockShap(country, region, sector, scheme, budget, duration) {
  const factors = [];
  const seed = hashCode(country + sector + scheme);

  // 国/地域
  const countryShap = {
    "サブサハラアフリカ": 0.22,
    "南アジア": 0.15,
    "東南アジア": 0.10,
    "中南米": 0.05,
    "中東・北アフリカ": 0.12
  };
  const countryVal = round2((countryShap[region] || 0.08) + seededRandom(seed) * 0.06 - 0.03);
  factors.push({
    factor: `国: ${country}`,
    value: countryVal,
    icon: "🌍",
    direction: countryVal > 0 ? "risk" : "safe",
    explanation: getCountryExplanation(country, region)
  });

  // セクター
  const sectorShap = {
    "運輸交通": 0.11, "水資源・防災": 0.08, "ガバナンス": 0.09,
    "資源・エネルギー": 0.07, "農業・農村開発": 0.04,
    "都市開発・環境": 0.05, "保健医療": -0.03, "教育": -0.04, "情報通信技術": 0.02
  };
  const sectorVal = round2((sectorShap[sector] || 0.03) + seededRandom(seed + 1) * 0.04 - 0.02);
  factors.push({
    factor: `セクター: ${sector}`,
    value: sectorVal,
    icon: "🏗️",
    direction: sectorVal > 0 ? "risk" : "safe",
    explanation: getSectorExplanation(sector)
  });

  // スキーム
  const schemeShap = {
    "有償資金協力": 0.09, "無償資金協力": 0.03, "技術協力": -0.03
  };
  const schemeVal = round2(schemeShap[scheme] || 0);
  factors.push({
    factor: `スキーム: ${scheme}`,
    value: schemeVal,
    icon: "📋",
    direction: schemeVal > 0 ? "risk" : "safe",
    explanation: getSchemeExplanation(scheme)
  });

  // 事業費
  let budgetVal, budgetExp;
  if (budget > 100) {
    budgetVal = round2(0.12 + seededRandom(seed + 2) * 0.02 - 0.01);
    budgetExp = "大規模案件。管理コストとステークホルダー調整の複雑性が増大";
  } else if (budget > 50) {
    budgetVal = round2(0.06 + seededRandom(seed + 2) * 0.02 - 0.01);
    budgetExp = "中〜大規模。効率性管理に注意が必要";
  } else if (budget > 10) {
    budgetVal = round2(-0.02 + seededRandom(seed + 2) * 0.02 - 0.01);
    budgetExp = "中規模案件。管理が比較的容易で効率性が維持されやすい";
  } else {
    budgetVal = round2(-0.05 + seededRandom(seed + 2) * 0.02 - 0.01);
    budgetExp = "小〜中規模案件。管理コストが抑えられる傾向";
  }
  factors.push({
    factor: `事業費: ${budget}億円`,
    value: budgetVal,
    icon: "💰",
    direction: budgetVal > 0 ? "risk" : "safe",
    explanation: budgetExp
  });

  // 事業期間
  let durVal, durExp;
  if (duration > 72) {
    durVal = round2(0.10 + seededRandom(seed + 3) * 0.02 - 0.01);
    durExp = "長期事業。カウンターパート異動や外部環境変化のリスクが蓄積";
  } else if (duration > 48) {
    durVal = round2(0.05 + seededRandom(seed + 3) * 0.02 - 0.01);
    durExp = "やや長期。C/P異動リスクが累積する傾向";
  } else if (duration > 24) {
    durVal = round2(-0.02 + seededRandom(seed + 3) * 0.02 - 0.01);
    durExp = "標準的な事業期間。大きなリスク要因ではない";
  } else {
    durVal = round2(-0.04 + seededRandom(seed + 3) * 0.02 - 0.01);
    durExp = "短期事業。迅速な実施が見込める";
  }
  factors.push({
    factor: `事業期間: ${duration}ヶ月`,
    value: durVal,
    icon: "⏱️",
    direction: durVal > 0 ? "risk" : "safe",
    explanation: durExp
  });

  // 絶対値でソート
  factors.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return factors;
}

/**
 * 入力条件に応じて教訓データを選択
 */
function selectLessonData(region, sector) {
  if (region.includes("アフリカ") && sector === "水資源・防災") {
    return AFRICA_WATER_LESSONS;
  } else if (region === "南アジア" && sector === "教育") {
    return SOUTHASIA_EDUCATION_LESSONS;
  } else {
    return DEFAULT_LESSONS;
  }
}

/**
 * 入力条件に応じて類似案件データを選択
 */
function selectSimilarProjects(region, sector) {
  if (region.includes("アフリカ") && sector === "水資源・防災") {
    return SIMILAR_PROJECTS_WATER;
  } else if (region === "南アジア" && sector === "教育") {
    return SIMILAR_PROJECTS_EDUCATION;
  } else {
    return SIMILAR_PROJECTS_DEFAULT;
  }
}

/**
 * 評価項目別のダミー予測
 */
function generateSubRatings(riskProb) {
  // リスクが高いほど低いレーティング
  const getBaseRating = (prob, bias) => {
    const adjusted = prob + bias;
    if (adjusted < 40) return { rating: "③", label: "良好", cls: "rating-3" };
    if (adjusted < 65) return { rating: "②", label: "やや懸念", cls: "rating-2" };
    return { rating: "①", label: "要注意", cls: "rating-1" };
  };

  return [
    { name: "妥当性", ...getBaseRating(riskProb, -25) },
    { name: "有効性", ...getBaseRating(riskProb, -5) },
    { name: "効率性", ...getBaseRating(riskProb, 5) },
    { name: "持続性", ...getBaseRating(riskProb, 12) },
  ];
}

// ============================================================
// 説明文ヘルパー
// ============================================================
function getCountryExplanation(country, region) {
  const map = {
    "サブサハラアフリカ": "サブサハラアフリカ地域は制度・財務基盤が相対的に脆弱。持続性の課題が頻出",
    "南アジア": "南アジア地域。行政能力にばらつきがあり、事業管理体制の構築が課題になりやすい",
    "東南アジア": "東南アジア地域。CLMV諸国では制度基盤の発展途上にあり、先発ASEAN諸国と差がある",
    "中南米": "中南米地域。制度基盤は比較的整備されているが、政権交代による政策変更リスクあり",
    "中東・北アフリカ": "中東・北アフリカ地域。治安・政情の変動リスクが事業実施に影響する可能性"
  };
  return map[region] || "対象国の制度・実施環境が評価に影響する傾向";
}

function getSectorExplanation(sector) {
  const map = {
    "運輸交通": "インフラ完成後の維持管理に経常的コストが発生。持続性の課題が出やすいセクター",
    "水資源・防災": "施設の維持管理コストと料金徴収体制の構築が持続性の鍵",
    "保健医療": "ソフト面中心の支援が多く、比較的安定した評価が得られる傾向",
    "教育": "人材育成効果が長期的に波及しやすく、評価が安定する傾向",
    "農業・農村開発": "自然環境・市場条件の変動影響を受けやすい",
    "ガバナンス": "制度改革の成果が可視化しにくく、有効性の評価が難しい傾向",
    "資源・エネルギー": "大規模インフラが多く、コスト超過・工期遅延のリスクあり",
    "都市開発・環境": "複数ステークホルダーの調整が必要で、効率性に課題が出やすい",
    "情報通信技術": "技術革新が速く、事業完了時に陳腐化するリスク"
  };
  return map[sector] || "";
}

function getSchemeExplanation(scheme) {
  const map = {
    "技術協力": "ソフト面の支援が中心。管理しやすく、評価が比較的安定する傾向",
    "有償資金協力": "大規模インフラが多い。コスト超過・工期遅延のリスクが相対的に高い",
    "無償資金協力": "施設建設が中心。維持管理の引き渡し後の持続性が課題になりやすい"
  };
  return map[scheme] || "";
}

// ============================================================
// ユーティリティ
// ============================================================
function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const c = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + c;
    hash |= 0;
  }
  return Math.abs(hash);
}

function seededRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function round2(n) {
  return Math.round(n * 100) / 100;
}
