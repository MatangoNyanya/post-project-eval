// ==============================================
// EvalNavi - メインアプリケーション
// ==============================================

// アプリ状態
const appState = {
  step: 1,
  inputData: {},
  riskScore: null,
  shapFactors: []
};

// ============================================================
// 初期化
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  initFormEvents();
  initNavEvents();
  initSidebar();
  updateSubsectors('運輸交通');
  updateRegion();
});

// ============================================================
// フォームイベント
// ============================================================
function initFormEvents() {
  // 国名変更 → 地域自動更新
  document.getElementById('countrySelect').addEventListener('change', updateRegion);

  // セクター変更 → サブセクター更新
  document.getElementById('sectorSelect').addEventListener('change', function() {
    updateSubsectors(this.value);
  });

  // スライダー → 値更新
  const slider = document.getElementById('durationSlider');
  slider.addEventListener('input', function() {
    updateSliderDisplay(parseInt(this.value));
    updateSliderGradient(this);
  });
  updateSliderGradient(slider);

  // スクリーニング実行ボタン
  document.getElementById('runScreeningBtn').addEventListener('click', runScreening);

  // 教訓確認ボタン
  document.getElementById('goToLessonsBtn').addEventListener('click', goToLessons);

  // 再分析ボタン
  document.getElementById('resetBtn').addEventListener('click', resetToStep1);
}

function updateRegion() {
  const country = document.getElementById('countrySelect').value;
  const region = COUNTRY_REGION[country] || '不明';
  document.getElementById('regionInput').value = region;
}

function updateSubsectors(sector) {
  const subsectorSel = document.getElementById('subsectorSelect');
  const options = SUBSECTORS[sector] || [];
  subsectorSel.innerHTML = options.map(s => `<option value="${s}">${s}</option>`).join('');
}

function updateSliderDisplay(val) {
  document.getElementById('durationValue').textContent = val;
  const years = Math.floor(val / 12);
  const months = val % 12;
  let hint = '';
  if (years > 0 && months > 0) hint = `（${years}年${months}ヶ月）`;
  else if (years > 0) hint = `（${years}年）`;
  else hint = '';
  document.getElementById('durationHint').textContent = hint;
}

function updateSliderGradient(slider) {
  const min = parseInt(slider.min);
  const max = parseInt(slider.max);
  const val = parseInt(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  slider.style.background = `linear-gradient(to right, #003A78 ${pct}%, #E0E4EA ${pct}%)`;
}

// ============================================================
// ナビゲーション
// ============================================================
function initNavEvents() {}

function goToStep(stepNum) {
  // ステップ表示切り替え
  document.querySelectorAll('.step-section').forEach(s => s.classList.add('hidden'));
  document.getElementById('step' + stepNum).classList.remove('hidden');

  // ステップインジケーター更新
  for (let i = 1; i <= 3; i++) {
    const circle = document.getElementById('step-circle-' + i);
    const nameEl = document.getElementById('step-label-' + i).querySelector('.step-name');

    if (i < stepNum) {
      circle.classList.remove('active');
      circle.classList.add('done');
      circle.innerHTML = '✓';
      nameEl.classList.add('active');
    } else if (i === stepNum) {
      circle.classList.add('active');
      circle.classList.remove('done');
      circle.innerHTML = String(i);
      nameEl.classList.add('active');
    } else {
      circle.classList.remove('active', 'done');
      circle.innerHTML = String(i);
      nameEl.classList.remove('active');
    }

    // ラインの更新
    if (i < 3) {
      const line = document.getElementById('step-line-' + i);
      if (i < stepNum) {
        line.classList.add('done');
      } else {
        line.classList.remove('done');
      }
    }
  }

  appState.step = stepNum;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================
// STEP 1 → 2: スクリーニング実行
// ============================================================
function runScreening() {
  // 入力値収集
  const country = document.getElementById('countrySelect').value;
  const region = document.getElementById('regionInput').value;
  const sector = document.getElementById('sectorSelect').value;
  const subsector = document.getElementById('subsectorSelect').value;
  const scheme = document.querySelector('input[name="scheme"]:checked').value;
  const budget = parseFloat(document.getElementById('budgetInput').value) || 8.5;
  const duration = parseInt(document.getElementById('durationSlider').value);
  const overview = document.getElementById('overviewText').value;

  appState.inputData = { country, region, sector, subsector, scheme, budget, duration, overview };

  // ローディング表示
  showLoading('過去の類似案件を分析中...');
  animateProgress(2000, () => {
    hideLoading();
    buildStep2();
    goToStep(2);
  });
}

// ============================================================
// STEP 2 構築
// ============================================================
function buildStep2() {
  const { country, region, sector, subsector, scheme, budget, duration } = appState.inputData;

  // リスクスコア計算
  const riskScore = calculateMockRisk(country, region, sector, scheme, budget, duration);
  appState.riskScore = riskScore;

  // SHAP生成
  const shapFactors = generateMockShap(country, region, sector, scheme, budget, duration);
  appState.shapFactors = shapFactors;

  // 入力サマリーバッジ
  renderInputSummary();

  // ゲージ
  renderGaugeChart(riskScore);

  // 判定テキスト
  renderVerdict(riskScore);

  // サブレーティング
  renderSubRatings(riskScore);

  // SHAPチャート
  renderShapChart(shapFactors);

  // SHAP説明テキスト
  renderShapExplanations(shapFactors);

  // 類似案件テーブル
  renderSimilarProjects();
}

function renderInputSummary() {
  const { country, sector, subsector, scheme, budget, duration } = appState.inputData;
  const container = document.getElementById('inputSummaryBadges');
  const badges = [
    `🌍 ${country}`, `🏗️ ${sector}（${subsector}）`,
    `📋 ${scheme}`, `💰 ${budget}億円`, `⏱️ ${duration}ヶ月`
  ];
  container.innerHTML = '<div class="input-summary-badges">' +
    badges.map(b => `<span class="input-badge">${b}</span>`).join('') +
    '</div>';
}

function renderVerdict(riskScore) {
  const el = document.getElementById('gaugeVerdict');
  if (riskScore < 40) {
    el.className = 'gauge-verdict verdict-good';
    el.textContent = '✅ 良好な評価が見込まれます';
  } else if (riskScore < 60) {
    el.className = 'gauge-verdict verdict-warning';
    el.textContent = '⚠️ 一部リスクが予想されます';
  } else {
    el.className = 'gauge-verdict verdict-danger';
    el.textContent = '🔴 要注意：重点的なリスク対策を推奨';
  }
}

function renderSubRatings(riskScore) {
  const ratings = generateSubRatings(riskScore);
  const container = document.getElementById('subRatingsContainer');
  container.innerHTML = ratings.map(r => `
    <div class="rating-card">
      <div class="rating-name">${r.name}</div>
      <div class="rating-value ${r.cls}">${r.rating}</div>
      <div class="rating-label">${r.label}</div>
    </div>
  `).join('');
}

function renderShapExplanations(shapFactors) {
  const container = document.getElementById('shapExplanations');
  container.innerHTML = shapFactors.map(f => `
    <div class="shap-explanation-item">
      <div class="shap-icon">${f.icon}</div>
      <div style="flex:1">
        <div class="shap-factor-name">${f.factor}</div>
        <div class="shap-exp-text">${f.explanation}</div>
      </div>
      <span class="${f.direction === 'risk' ? 'shap-badge-risk' : 'shap-badge-safe'}">
        ${f.direction === 'risk' ? '▲ リスク' : '▼ 低減'}
      </span>
    </div>
  `).join('');
}

function renderSimilarProjects() {
  const { region, sector } = appState.inputData;
  const projects = selectSimilarProjects(region, sector);

  // 統計集計
  const counts = { A: 0, B: 0, C: 0, D: 0 };
  projects.forEach(p => { if (counts[p.総合評価] !== undefined) counts[p.総合評価]++; });
  const total = projects.length;

  const statsEl = document.getElementById('tableSummaryStats');
  statsEl.innerHTML = `<span style="font-size:0.78rem;color:#6B7280">類似案件${total}件中：</span> ` +
    ['A', 'B', 'C', 'D'].map(g =>
      `<span class="stat-chip stat-${g}">${g}=${counts[g]}件 (${Math.round(counts[g]/total*100)}%)</span>`
    ).join('');

  const tbody = document.getElementById('similarProjectsTbody');
  tbody.innerHTML = projects.map(p => {
    const evalClass = `eval-${p.総合評価}`;
    const eff = p.効率性.replace('①', '①').replace('②', '②').replace('③', '③');
    const effNum = p.効率性.includes('③') ? 3 : p.効率性.includes('②') ? 2 : 1;
    const sustNum = p.持続性.includes('③') ? 3 : p.持続性.includes('②') ? 2 : 1;
    return `
      <tr>
        <td style="max-width:280px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${p.事業名}">${p.事業名}</td>
        <td>${p.評価年}</td>
        <td>${p.セクター}</td>
        <td><span class="eval-badge ${evalClass}">${p.総合評価}</span></td>
        <td class="rating-cell-${effNum}">${p.効率性}</td>
        <td class="rating-cell-${sustNum}">${p.持続性}</td>
      </tr>
    `;
  }).join('');
}

// ============================================================
// STEP 2 → 3: 教訓確認
// ============================================================
function goToLessons() {
  showLoading('過去の評価書から関連する教訓を検索中...');
  animateProgress(3000, () => {
    hideLoading();
    buildStep3();
    goToStep(3);
  });
}

// ============================================================
// STEP 3 構築
// ============================================================
function buildStep3() {
  const { region, sector } = appState.inputData;
  const lessonData = selectLessonData(region, sector);

  renderLessonPanels(lessonData);
  renderSummary(lessonData.summary);
}

function renderLessonPanels(lessonData) {
  const container = document.getElementById('lessonPanels');
  container.innerHTML = '';

  lessonData.risk_factors.forEach((rf, idx) => {
    const panel = buildLessonPanel(rf, idx);
    container.appendChild(panel);
  });

  // 最初のパネルを開く
  const firstHeader = container.querySelector('.lesson-panel-header');
  if (firstHeader) {
    toggleLessonPanel(firstHeader);
  }
}

function buildLessonPanel(rf, idx) {
  const panel = document.createElement('div');
  panel.className = 'lesson-panel';

  // ヘッダー
  const header = document.createElement('div');
  header.className = 'lesson-panel-header';
  header.innerHTML = `
    <div class="lesson-panel-title">
      <span class="lesson-rank-badge">${rf.shap_rank}</span>
      <span>${rf.factor_icon} ${rf.factor_name}</span>
    </div>
    <div class="lesson-meta">
      <span class="lesson-shap-val">SHAP ${rf.shap_value}</span>
      <span class="lesson-count-val">類似事業 ${rf.similar_count}件</span>
      <span class="lesson-toggle">▼</span>
    </div>
  `;
  header.addEventListener('click', () => toggleLessonPanel(header));

  // ボディ
  const body = document.createElement('div');
  body.className = 'lesson-panel-body';

  const inner = document.createElement('div');
  inner.className = 'lesson-inner';

  // 教訓
  rf.key_lessons.forEach((lesson, li) => {
    const item = document.createElement('div');
    item.className = 'lesson-item';

    const sources = lesson.sources.map(s =>
      `<span class="source-tag" title="${s.name}（${s.year}年）${s.page}">${s.name}（${s.year}）${s.page}</span>`
    ).join('');

    item.innerHTML = `
      <div class="lesson-subtitle">🔑 教訓 ${rf.shap_rank}-${li + 1}：${lesson.title}</div>
      <div class="lesson-quote">${lesson.text}</div>
      <div class="lesson-sources">
        <span class="source-label">📎 出典:</span>
        ${sources}
      </div>
    `;

    if (li < rf.key_lessons.length - 1) {
      item.innerHTML += '<hr class="lesson-divider">';
    }

    inner.appendChild(item);
  });

  // 成功パターン
  const successDiv = document.createElement('div');
  successDiv.className = 'success-patterns';
  successDiv.innerHTML = `
    <div class="success-title">✅ 成功事業で見られた対策パターン</div>
    <ul class="success-list">
      ${rf.success_patterns.map(p => `<li>${p}</li>`).join('')}
    </ul>
  `;
  inner.appendChild(successDiv);

  // 統計（あれば）
  if (rf.statistics) {
    const statsDiv = document.createElement('div');
    statsDiv.className = 'stats-box';
    statsDiv.innerHTML = `
      <div class="stats-box-title">📊 統計的補足：${rf.statistics.description}</div>
      ${rf.statistics.data.map(d => `
        <div class="stats-bar-item">
          <div class="stats-bar-label">
            <span class="stats-bar-name">${d.label}</span>
            <span class="stats-bar-pct">${d.value}%</span>
          </div>
          <div class="stats-bar-track">
            <div class="stats-bar-fill ${d.colorClass}" style="width: 0%" data-width="${d.value}%"></div>
          </div>
          <div class="stats-bar-detail">${d.detail}</div>
        </div>
      `).join('')}
    `;
    inner.appendChild(statsDiv);
  }

  body.appendChild(inner);
  panel.appendChild(header);
  panel.appendChild(body);

  return panel;
}

function toggleLessonPanel(header) {
  const body = header.nextElementSibling;
  const toggle = header.querySelector('.lesson-toggle');
  const isOpen = body.classList.contains('open');

  if (isOpen) {
    body.classList.remove('open');
    header.classList.remove('open');
    toggle.classList.remove('open');
  } else {
    body.classList.add('open');
    header.classList.add('open');
    toggle.classList.add('open');

    // 統計バーのアニメーション
    setTimeout(() => {
      body.querySelectorAll('.stats-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width;
      });
    }, 100);
  }
}

function renderSummary(summary) {
  document.getElementById('summaryHeader').textContent = summary.title;
  const pointsEl = document.getElementById('summaryPoints');
  pointsEl.innerHTML = summary.points.map(p => `
    <div class="summary-point-item">
      <div class="summary-num">${p.number}</div>
      <div class="summary-point-content">
        <div class="summary-point-text">${p.text}</div>
        <div class="summary-related">💬 関連要因: ${p.related_factor}</div>
      </div>
    </div>
  `).join('');
}

// ============================================================
// STEP 3 → 1: リセット
// ============================================================
function resetToStep1() {
  goToStep(1);
}

// ============================================================
// ローディング
// ============================================================
function showLoading(text) {
  document.getElementById('loadingText').textContent = text;
  document.getElementById('loadingProgressBar').style.width = '0%';
  document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
  document.getElementById('loadingOverlay').classList.remove('active');
}

function animateProgress(durationMs, callback) {
  const bar = document.getElementById('loadingProgressBar');
  let progress = 0;
  const interval = 60;
  const steps = durationMs / interval;
  const increment = 92 / steps;

  const timer = setInterval(() => {
    progress += increment + Math.random() * increment * 0.5;
    if (progress >= 92) {
      progress = 92;
      clearInterval(timer);
    }
    bar.style.width = progress + '%';
  }, interval);

  setTimeout(() => {
    clearInterval(timer);
    bar.style.width = '100%';
    setTimeout(callback, 150);
  }, durationMs);
}

// ============================================================
// サイドバー
// ============================================================
function initSidebar() {
  const toggle = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  const closeBtn = document.getElementById('sidebarClose');

  toggle.addEventListener('click', () => {
    sidebar.classList.add('open');
    overlay.classList.add('active');
  });
  overlay.addEventListener('click', closeSidebar);
  closeBtn.addEventListener('click', closeSidebar);
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('active');
}
