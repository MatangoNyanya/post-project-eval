// ==============================================
// EvalNavi - ECharts グラフ描画
// ==============================================

let gaugeChartInstance = null;
let shapChartInstance = null;

/**
 * ゲージチャート（リスク確率）
 */
function renderGaugeChart(riskProbability) {
  const el = document.getElementById('gaugeChart');
  if (!el) return;

  if (gaugeChartInstance) {
    gaugeChartInstance.dispose();
  }
  gaugeChartInstance = echarts.init(el);

  let gaugeColor;
  if (riskProbability < 40) gaugeColor = '#2E7D32';
  else if (riskProbability < 60) gaugeColor = '#FF6D00';
  else gaugeColor = '#D32F2F';

  const option = {
    series: [
      {
        type: 'gauge',
        startAngle: 180,
        endAngle: 0,
        min: 0,
        max: 100,
        splitNumber: 5,
        radius: '90%',
        center: ['50%', '68%'],
        axisLine: {
          lineStyle: {
            width: 18,
            color: [
              [0.4, '#C8E6C9'],
              [0.6, '#FFE0B2'],
              [1, '#FFCDD2']
            ]
          }
        },
        pointer: {
          icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
          length: '60%',
          width: 12,
          offsetCenter: [0, '-55%'],
          itemStyle: {
            color: gaugeColor
          }
        },
        axisTick: {
          length: 8,
          lineStyle: { color: 'auto', width: 1 }
        },
        splitLine: {
          length: 16,
          lineStyle: { color: 'auto', width: 2 }
        },
        axisLabel: {
          color: '#6B7280',
          fontSize: 11,
          distance: -36,
          formatter: function(val) {
            if (val === 0 || val === 20 || val === 40 || val === 60 || val === 80 || val === 100) {
              return val + '%';
            }
            return '';
          }
        },
        title: {
          offsetCenter: [0, '-16%'],
          fontSize: 12,
          color: '#6B7280',
          fontFamily: 'Noto Sans JP'
        },
        detail: {
          fontSize: 40,
          offsetCenter: [0, '16%'],
          valueAnimation: true,
          formatter: function(value) {
            return Math.round(value) + '%';
          },
          color: gaugeColor,
          fontFamily: 'Inter, Noto Sans JP',
          fontWeight: '700'
        },
        data: [
          {
            value: riskProbability,
            name: '要注意確率'
          }
        ]
      }
    ]
  };

  gaugeChartInstance.setOption(option);

  // レスポンシブ
  window.addEventListener('resize', () => {
    if (gaugeChartInstance) gaugeChartInstance.resize();
  });
}

/**
 * SHAPバーチャート（横棒）
 */
function renderShapChart(shapFactors) {
  const el = document.getElementById('shapChart');
  if (!el) return;

  if (shapChartInstance) {
    shapChartInstance.dispose();
  }
  shapChartInstance = echarts.init(el);

  const labels = shapFactors.map(f => f.icon + ' ' + f.factor);
  const values = shapFactors.map(f => f.value);
  const colors = shapFactors.map(f => f.direction === 'risk' ? '#FF6D00' : '#2E7D32');

  const option = {
    grid: {
      left: '2%',
      right: '12%',
      top: '5%',
      bottom: '5%',
      containLabel: true
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: function(params) {
        const d = params[0];
        const sign = d.value >= 0 ? '+' : '';
        const color = d.value >= 0 ? '#FF6D00' : '#2E7D32';
        return `<div style="font-size:12px;font-family:Noto Sans JP">
          <strong>${d.name}</strong><br>
          SHAP値: <strong style="color:${color}">${sign}${d.value.toFixed(2)}</strong>
        </div>`;
      }
    },
    xAxis: {
      type: 'value',
      axisLabel: {
        formatter: function(val) {
          return (val >= 0 ? '+' : '') + val.toFixed(2);
        },
        fontSize: 11,
        color: '#6B7280'
      },
      splitLine: {
        lineStyle: { color: '#F0F0F0', type: 'dashed' }
      },
      axisLine: {
        show: true,
        lineStyle: { color: '#E0E4EA' }
      }
    },
    yAxis: {
      type: 'category',
      data: labels,
      axisLabel: {
        fontSize: 11,
        color: '#374151',
        fontFamily: 'Noto Sans JP'
      },
      axisLine: { show: false },
      axisTick: { show: false }
    },
    series: [
      {
        type: 'bar',
        data: values.map((v, i) => ({
          value: v,
          itemStyle: {
            color: colors[i],
            borderRadius: v >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4],
            opacity: 0.85
          }
        })),
        label: {
          show: true,
          position: function(params) {
            return params.value >= 0 ? 'right' : 'left';
          },
          formatter: function(params) {
            const sign = params.value >= 0 ? '+' : '';
            return sign + params.value.toFixed(2);
          },
          fontSize: 11,
          fontWeight: '700',
          color: function(params) {
            return params.value >= 0 ? '#FF6D00' : '#2E7D32';
          }
        },
        barMaxWidth: 28
      }
    ]
  };

  shapChartInstance.setOption(option);

  window.addEventListener('resize', () => {
    if (shapChartInstance) shapChartInstance.resize();
  });
}

/**
 * リサイズ対応
 */
function resizeAllCharts() {
  if (gaugeChartInstance) gaugeChartInstance.resize();
  if (shapChartInstance) shapChartInstance.resize();
}
