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
  return (
    <div className="bars-wrap">
      {items.map((item) => {
        const label = item[labelKey];
        const value = Number(item[valueKey] || 0);
        const width = `${(value / max) * 100}%`;
        const highlighted = highlightLabel && label === highlightLabel;
        return (
          <div className={`bar-row ${highlighted ? 'highlighted' : ''}`} key={label}>
            <span className="bar-label">{label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width }} />
            </div>
            <span className="bar-value">{fmt(value, value > 1 ? 2 : 4)}</span>
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
    </div>
  );
}

function PcaScatter({ data }) {
  const points = useMemo(() => {
    if (!Array.isArray(data) || data.length === 0) return [];
    const sample = data.length > 500 ? data.slice(0, 500) : data;

    const xs = sample.map((p) => p.pc1);
    const ys = sample.map((p) => p.pc2);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const genreList = [...new Set(sample.map((p) => p.genre))];
    const palette = ['#1b4332', '#2d6a4f', '#40916c', '#52b788', '#74c69d', '#95d5b2', '#388e5f', '#276749'];
    const colorByGenre = Object.fromEntries(genreList.map((g, idx) => [g, palette[idx % palette.length]]));

    return sample.map((p, idx) => {
      const x = ((p.pc1 - minX) / (maxX - minX || 1)) * 760 + 20;
      const y = 300 - ((p.pc2 - minY) / (maxY - minY || 1)) * 260 + 20;
      return {
        id: `${p.genre}-${idx}`,
        x,
        y,
        genre: p.genre,
        color: colorByGenre[p.genre]
      };
    });
  }, [data]);

  if (points.length === 0) {
    return <p className="muted">PCA indisponível no momento.</p>;
  }

  return (
    <div className="pca-scatter">
      <svg viewBox="0 0 800 340" role="img" aria-label="PCA 2D">
        <rect x="0" y="0" width="800" height="340" fill="rgba(255,255,255,0.5)" rx="8" />
        {points.map((pt) => (
          <circle key={pt.id} cx={pt.x} cy={pt.y} r="3.3" fill={pt.color} opacity="0.72" />
        ))}
      </svg>
      <p className="muted">Cada ponto é uma faixa do dataset (amostra). Cores representam gêneros.</p>
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
  const [pcaData, setPcaData] = useState(null);

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
        const [infoRes, distRes, compareRes, cmRes, importanceRes, pcaRes] = await Promise.allSettled([
          fetchJson('/api/model-info'),
          fetchJson('/api/charts/genre-distribution'),
          fetchJson('/api/charts/model-comparison'),
          fetchJson('/api/charts/confusion-matrix'),
          fetchJson('/api/charts/feature-importance'),
          fetchJson('/api/charts/pca')
        ]);

        if (!mounted) return;

        if (infoRes.status === 'fulfilled') setModelInfo(infoRes.value);
        if (distRes.status === 'fulfilled') setGenreDistribution(distRes.value.data || []);
        if (compareRes.status === 'fulfilled') setModelComparison(compareRes.value.data || []);
        if (cmRes.status === 'fulfilled') setConfusionData(cmRes.value);
        if (importanceRes.status === 'fulfilled') setFeatureImportance(importanceRes.value);
        if (pcaRes.status === 'fulfilled') setPcaData(pcaRes.value);

        const errors = [infoRes, distRes, compareRes, cmRes, importanceRes, pcaRes]
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
        <p className="eyebrow">Aprendizado de Máquina I</p>
        <h1>Classificador de Gênero Musical</h1>
        <p>
          Fluxo completo com FMA: treino supervisionado com features pré-computadas (metadata), inferência em
          MP3 novo com extração compatível e painéis para análise comparativa.
        </p>
      </header>

      <main className="grid">
        <Section
          title="Upload e predição"
          subtitle="Envie um .mp3 ou .wav para o backend FastAPI extrair atributos e prever o gênero"
        >
          <form className="upload-form" onSubmit={onPredict}>
            <input
              type="file"
              accept=".mp3,.wav,audio/mpeg,audio/wav"
              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
            />
            <button type="submit" disabled={!selectedFile || predicting}>
              {predicting ? 'Prevendo...' : 'Prever gênero'}
            </button>
          </form>

          {predictionError ? <p className="error">{predictionError}</p> : null}

          {prediction ? (
            <div className="prediction-box">
              <p className="muted">Arquivo: {prediction.filename}</p>
              <h3>{prediction.predicted_genre}</h3>
              <p className="muted">Modelo: {prediction.model_name}</p>
              {prediction.notes?.length ? (
                <ul className="notes-list">
                  {prediction.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          ) : null}

          {prediction?.probabilities?.length ? (
            <>
              <h4>Ranking de probabilidades</h4>
              <HorizontalBars items={prediction.probabilities} labelKey="genre" valueKey="probability" />
            </>
          ) : null}
        </Section>

        <Section
          title="Comparação do arquivo com o gênero previsto"
          subtitle="Relação entre o áudio enviado e os padrões médios observados no dataset"
        >
          {prediction?.feature_comparison?.length ? (
            <div className="comparison-table-wrap">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Valor do arquivo</th>
                    <th>Média do gênero previsto</th>
                    <th>Média global</th>
                  </tr>
                </thead>
                <tbody>
                  {prediction.feature_comparison.map((item) => (
                    <tr key={item.feature}>
                      <td>{item.feature}</td>
                      <td>{fmt(item.uploaded_value, 3)}</td>
                      <td>{fmt(item.predicted_genre_mean, 3)}</td>
                      <td>{fmt(item.global_mean, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">Envie um arquivo para ver a comparação detalhada.</p>
          )}

          {prediction?.extracted_features ? (
            <>
              <h4>Features com maior variação no upload</h4>
              <HorizontalBars
                items={Object.entries(prediction.extracted_features).map(([feature, value]) => ({
                  feature,
                  value
                }))}
                labelKey="feature"
                valueKey="value"
              />
            </>
          ) : null}
        </Section>

        <Section title="Distribuição dos gêneros" subtitle="Base de treinamento atual (features pré-computadas do metadata)">
          {loadingDashboard ? <p className="muted">Carregando...</p> : null}
          {genreDistribution.length > 0 ? (
            <HorizontalBars
              items={genreDistribution}
              labelKey="genre"
              valueKey="count"
              highlightLabel={predictedGenre}
            />
          ) : (
            <p className="muted">Distribuição indisponível. Rode `python src/extract_features.py`.</p>
          )}
        </Section>

        <Section title="Comparação entre modelos" subtitle="Decision Tree vs KNN vs Random Forest">
          {modelComparison.length > 0 ? (
            <div className="cards-metrics">
              {modelComparison.map((item) => (
                <article className="metric-card" key={item.model}>
                  <h4>{item.model}</h4>
                  <p>Accuracy: {fmt(item.metrics.accuracy)}</p>
                  <p>Precision Macro: {fmt(item.metrics.precision_macro)}</p>
                  <p>Recall Macro: {fmt(item.metrics.recall_macro)}</p>
                  <p>F1 Macro: {fmt(item.metrics.f1_macro)}</p>
                </article>
              ))}
            </div>
          ) : (
            <p className="muted">Comparação indisponível. Rode `python src/train.py`.</p>
          )}

          {modelInfo?.metrics_summary?.best_model ? (
            <p className="model-summary">
              Melhor modelo salvo: <strong>{modelInfo.metrics_summary.best_model}</strong>
            </p>
          ) : null}
        </Section>

        <Section title="Matriz de confusão" subtitle="Desempenho do melhor modelo">
          {confusionData?.matrix?.length ? (
            <ConfusionHeatmap labels={confusionData.labels} matrix={confusionData.matrix} />
          ) : (
            <p className="muted">Matriz indisponível.</p>
          )}
        </Section>

        <Section title="Importância dos atributos (RF)" subtitle="Top features quando o melhor modelo é Random Forest">
          {featureImportance?.available ? (
            <HorizontalBars items={featureImportance.data} labelKey="feature" valueKey="importance" />
          ) : (
            <p className="muted">{featureImportance?.reason || 'Importância indisponível.'}</p>
          )}
        </Section>

        <Section title="PCA 2D" subtitle="Visualização da estrutura dos dados no espaço de features">
          <PcaScatter data={pcaData?.data || []} />
          {pcaData?.explained_variance_ratio ? (
            <p className="muted">
              Variância explicada: PC1={fmt(pcaData.explained_variance_ratio[0], 3)}, PC2={fmt(pcaData.explained_variance_ratio[1], 3)}
            </p>
          ) : null}
        </Section>

        <Section
          title="Treinamento vs Demonstração"
          subtitle="Como treinamento e demonstração se conectam na apresentação"
        >
          <div className="approach-grid">
            <article>
              <h4>Treinamento (features pré-computadas)</h4>
              <p>
                É o caminho principal do treino atual, com features prontas do `features.csv`, rápido e estável para
                modelagem.
              </p>
            </article>
            <article>
              <h4>Demonstração (extração no upload)</h4>
              <p>
                O áudio enviado é transformado em features compatíveis com o esquema do modelo treinado, permitindo
                testar MP3 novo no mesmo classificador.
              </p>
            </article>
          </div>

          <div className="images-grid">
            <div>
              <h5>Gráfico salvo: Matriz de confusão (imagem)</h5>
              <img src={buildUrl('/assets/confusion_matrix_best_model.png')} alt="Matriz de confusão" />
            </div>
            <div>
              <h5>Gráfico salvo: Comparação de métricas (imagem)</h5>
              <img src={buildUrl('/assets/model_comparison_metrics.png')} alt="Comparação de métricas" />
            </div>
            <div>
              <h5>Gráfico salvo: Importância de features (imagem)</h5>
              <img src={buildUrl('/assets/random_forest_feature_importance.png')} alt="Importância de features" />
            </div>
          </div>
        </Section>

        {dashboardError ? <p className="error">Avisos de dados/API: {dashboardError}</p> : null}
      </main>
    </div>
  );
}
