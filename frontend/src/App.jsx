import { useEffect, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || '';

const buildUrl = (path) => {
  if (!API_BASE) return path;
  return `${API_BASE}${path}`;
};

const fmt = (value, digits = 4) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '-';
  }
  return Number(value).toFixed(digits);
};

async function fetchJson(path, options) {
  const response = await fetch(buildUrl(path), options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Erro ${response.status} em ${path}`);
  }
  return response.json();
}

function Section({ title, subtitle, children }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {subtitle ? <p className="panel-subtitle">{subtitle}</p> : null}
      {children}
    </section>
  );
}

function HorizontalBars({ items, labelKey, valueKey, highlightLabel }) {
  const max = Math.max(...items.map((i) => Number(i[valueKey] || 0)), 1e-9);
  const total = items.reduce((acc, cur) => acc + Number(cur[valueKey] || 0), 0);
  return (
    <div className="bars-wrap">
      {items.map((item) => {
        const label = item[labelKey];
        const value = Number(item[valueKey] || 0);
        const width = `${(value / max) * 100}%`;
        const highlighted = highlightLabel && label === highlightLabel;
        const pct = total > 0 ? (value / total) * 100 : 0;
        return (
          <div className={`bar-row ${highlighted ? 'highlighted' : ''}`} key={label}>
            <span className="bar-label">{label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width }} />
            </div>
            <span className="bar-value">{fmt(value, value > 1 ? 0 : 4)} ({fmt(pct, 1)}%)</span>
          </div>
        );
      })}
    </div>
  );
}

function ConfusionHeatmap({ labels, matrix }) {
  if (!labels || !matrix) return null;
  const flat = matrix.flat();
  const max = Math.max(...flat, 1);

  return (
    <div className="heatmap-wrap">
      <table className="heatmap-table">
        <thead>
          <tr>
            <th>Real \\ Pred</th>
            {labels.map((label) => (
              <th key={`h-${label}`}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIdx) => (
            <tr key={`r-${labels[rowIdx]}`}>
              <th>{labels[rowIdx]}</th>
              {row.map((value, colIdx) => {
                const intensity = value / max;
                return (
                  <td
                    key={`${rowIdx}-${colIdx}`}
                    style={{ backgroundColor: `rgba(30, 110, 65, ${0.08 + 0.85 * intensity})` }}
                  >
                    {value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted">Generos: {labels.map((label, idx) => `${idx + 1}. ${label}`).join(' | ')}</p>
    </div>
  );
}

function EmbeddingScatter({ data, xKey = 'x', yKey = 'y', title = 'Embedding 2D' }) {
  const { points, genres } = useMemo(() => {
    if (!Array.isArray(data) || data.length === 0) return { points: [], genres: [] };
    const sample = data.length > 500 ? data.slice(0, 500) : data;

    const xs = sample.map((p) => Number(p[xKey]));
    const ys = sample.map((p) => Number(p[yKey]));
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const genreList = [...new Set(sample.map((p) => p.genre))];
    const palette = ['#e63946', '#ff7f11', '#ffbe0b', '#2a9d8f', '#3a86ff', '#8338ec', '#f72585', '#4361ee'];
    const colorByGenre = Object.fromEntries(genreList.map((g, idx) => [g, palette[idx % palette.length]]));

    const mapped = sample.map((p, idx) => {
      const x = ((Number(p[xKey]) - minX) / (maxX - minX || 1)) * 760 + 20;
      const y = 300 - ((Number(p[yKey]) - minY) / (maxY - minY || 1)) * 260 + 20;
      return {
        id: `${p.genre}-${idx}`,
        x,
        y,
        genre: p.genre,
        color: colorByGenre[p.genre]
      };
    });

    return { points: mapped, genres: genreList.map((genre) => ({ genre, color: colorByGenre[genre] })) };
  }, [data, xKey, yKey]);

  if (points.length === 0) {
    return <p className="muted">Visualizacao indisponivel no momento.</p>;
  }

  return (
    <div className="pca-scatter">
      <svg viewBox="0 0 800 340" role="img" aria-label={title}>
        <rect x="0" y="0" width="800" height="340" fill="rgba(255,255,255,0.5)" rx="8" />
        {points.map((pt) => (
          <circle key={pt.id} cx={pt.x} cy={pt.y} r="4.2" fill={pt.color} opacity="0.84" stroke="#133022" strokeWidth="0.45" />
        ))}
      </svg>
      <div className="genre-legend">
        {genres.map((item) => (
          <span key={item.genre} className="legend-item">
            <i className="legend-dot" style={{ backgroundColor: item.color }} />
            {item.genre}
          </span>
        ))}
      </div>
      <p className="muted">Cada ponto e uma faixa. Cores diferentes representam os generos.</p>
    </div>
  );
}

export default function App() {
  const [loadingDashboard, setLoadingDashboard] = useState(true);
  const [dashboardError, setDashboardError] = useState('');

  const [modelInfo, setModelInfo] = useState(null);
  const [genreDistribution, setGenreDistribution] = useState([]);
  const [modelComparison, setModelComparison] = useState([]);
  const [confusionData, setConfusionData] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [embeddingData, setEmbeddingData] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState('');
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    let mounted = true;

    async function loadDashboard() {
      setLoadingDashboard(true);
      setDashboardError('');

      try {
        const [infoRes, distRes, compareRes, cmRes, importanceRes, ldaRes] = await Promise.allSettled([
          fetchJson('/api/model-info'),
          fetchJson('/api/charts/genre-distribution'),
          fetchJson('/api/charts/model-comparison'),
          fetchJson('/api/charts/confusion-matrix'),
          fetchJson('/api/charts/feature-importance'),
          fetchJson('/api/charts/lda')
        ]);

        if (!mounted) return;

        if (infoRes.status === 'fulfilled') setModelInfo(infoRes.value);
        if (distRes.status === 'fulfilled') setGenreDistribution(distRes.value.data || []);
        if (compareRes.status === 'fulfilled') setModelComparison(compareRes.value.data || []);
        if (cmRes.status === 'fulfilled') setConfusionData(cmRes.value);
        if (importanceRes.status === 'fulfilled') setFeatureImportance(importanceRes.value);
        if (ldaRes.status === 'fulfilled') setEmbeddingData(ldaRes.value);

        const errors = [infoRes, distRes, compareRes, cmRes, importanceRes]
          .filter((res) => res.status === 'rejected')
          .map((res) => res.reason?.message || 'Erro desconhecido');

        if (errors.length > 0) {
          setDashboardError(errors.join(' | '));
        }
      } catch (err) {
        if (!mounted) return;
        setDashboardError(err.message || String(err));
      } finally {
        if (mounted) setLoadingDashboard(false);
      }
    }

    loadDashboard();
    return () => {
      mounted = false;
    };
  }, []);

  async function onPredict(event) {
    event.preventDefault();
    if (!selectedFile) return;

    setPredicting(true);
    setPredictionError('');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const data = await fetchJson('/api/predict', {
        method: 'POST',
        body: formData
      });
      setPrediction(data);
    } catch (err) {
      setPredictionError(err.message || String(err));
      setPrediction(null);
    } finally {
      setPredicting(false);
    }
  }

  const predictedGenre = prediction?.predicted_genre;
  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">Aprendizado de Maquina I</p>
        <h1>Classificador de Genero Musical</h1>
        <p>Fluxo completo com FMA para treino supervisionado, inferencia em MP3 novo e paineis de analise.</p>
      </header>

      <main className="grid">
        <Section title="Upload e predicao" subtitle="Envie um .mp3 ou .wav para prever o genero">
          <form className="upload-form" onSubmit={onPredict}>
            <input
              type="file"
              accept=".mp3,.wav,audio/mpeg,audio/wav"
              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
            />
            <button type="submit" disabled={!selectedFile || predicting}>
              {predicting ? 'Prevendo...' : 'Prever genero'}
            </button>
          </form>

          {predictionError ? <p className="error">{predictionError}</p> : null}

          {prediction ? (
            <div className="prediction-box">
              <p className="muted">Arquivo: {prediction.filename}</p>
              <h3>{prediction.predicted_genre}</h3>
              <p className="muted">Modelo: {prediction.model_name}</p>
            </div>
          ) : null}

          {prediction?.probabilities?.length ? (
            <>
              <h4>Ranking de probabilidades</h4>
              <HorizontalBars items={prediction.probabilities} labelKey="genre" valueKey="probability" />
            </>
          ) : null}
        </Section>

        <Section title="Distribuicao dos generos" subtitle="Base de treinamento atual">
          {loadingDashboard ? <p className="muted">Carregando...</p> : null}
          {genreDistribution.length > 0 ? (
            <HorizontalBars items={genreDistribution} labelKey="genre" valueKey="count" highlightLabel={predictedGenre} />
          ) : (
            <p className="muted">Distribuicao indisponivel.</p>
          )}
        </Section>

        <Section title="Comparacao entre modelos" subtitle="Decision Tree vs KNN vs Random Forest">
          {modelComparison.length > 0 ? (
            <div className="cards-metrics">
              {modelComparison.map((item) => (
                <article
                  className={`metric-card ${modelInfo?.metrics_summary?.best_model === item.model ? 'metric-card-best' : ''}`}
                  key={item.model}
                >
                  <h4>{item.model}</h4>
                  <p>Accuracy: {fmt(item.metrics.accuracy)}</p>
                  <p>Precision Macro: {fmt(item.metrics.precision_macro)}</p>
                  <p>Recall Macro: {fmt(item.metrics.recall_macro)}</p>
                  <p>F1 Macro: {fmt(item.metrics.f1_macro)}</p>
                </article>
              ))}
            </div>
          ) : (
            <p className="muted">Comparacao indisponivel.</p>
          )}

          {modelInfo?.metrics_summary?.best_model ? (
            <p className="model-summary">
              Melhor modelo salvo: <strong>{modelInfo.metrics_summary.best_model}</strong>
            </p>
          ) : null}
        </Section>

        <Section title="Matriz de confusao" subtitle="Desempenho do melhor modelo">
          {confusionData?.matrix?.length ? (
            <ConfusionHeatmap labels={confusionData.labels} matrix={confusionData.matrix} />
          ) : (
            <p className="muted">Matriz indisponivel.</p>
          )}
        </Section>

        <Section title="Importancia dos atributos (RF)" subtitle="Top features quando o melhor modelo e Random Forest">
          {featureImportance?.available ? (
            <HorizontalBars items={featureImportance.data} labelKey="feature" valueKey="importance" />
          ) : (
            <p className="muted">{featureImportance?.reason || 'Importancia indisponivel.'}</p>
          )}
        </Section>

        <Section title="Mapa de generos 2D (LDA)" subtitle="Projecao supervisionada para separar melhor os generos">
          <EmbeddingScatter data={embeddingData?.data || []} xKey="x" yKey="y" title={embeddingData?.chart || 'LDA 2D'} />
        </Section>

        <Section title="Graficos salvos" subtitle="Artefatos gerados no backend">
          <div className="images-grid">
            <div>
              <h5>Matriz de confusao</h5>
              <img src={buildUrl('/assets/confusion_matrix_best_model.png')} alt="Matriz de confusao" />
            </div>
            <div>
              <h5>Comparacao de metricas</h5>
              <img src={buildUrl('/assets/model_comparison_metrics.png')} alt="Comparacao de metricas" />
            </div>
            <div>
              <h5>Importancia de features</h5>
              <img src={buildUrl('/assets/random_forest_feature_importance.png')} alt="Importancia de features" />
            </div>
          </div>
        </Section>

        {dashboardError ? <p className="error">Avisos de dados/API: {dashboardError}</p> : null}
      </main>
    </div>
  );
}
