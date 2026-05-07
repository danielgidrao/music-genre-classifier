# Music Genre Classifier (FMA)

Projeto de classificação supervisionada multiclasse para prever gênero musical com:
- Decision Tree
- KNN
- Random Forest

## Estrutura

## 1) Dataset FMA

Repositório oficial: [mdeff/fma](https://github.com/mdeff/fma)

Baixe:
- `fma_small.zip`
- `fma_metadata.zip`

Organize assim:

```text
music-genre-classifier/
  data/
    raw/
      fma_small/
        000/
        001/
        ...
      fma_metadata/
        tracks.csv
        features.csv
        genres.csv
        echonest.csv
```

## 2) Rodar com Docker Compose (recomendado)

Pré-requisito: Docker Desktop instalado.

### Subir API + Frontend

```bash
docker compose up --build
```

Acessos:
- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`
- Docs Swagger: `http://localhost:8000/docs`

### Pipeline de ML via container

Fluxo recomendado (treino com metadata pronto):

```bash
docker compose run --rm api python src/train.py --training-source metadata
docker compose run --rm api python src/evaluate.py
docker compose run --rm api python src/predict.py caminho/para/musica.mp3
```

Se quiser gerar features diretamente dos áudios (`fma_small`) por comparação:

```bash
docker compose run --rm api python src/extract_features.py --feature-mode basic
docker compose run --rm api python src/extract_features.py --feature-mode fma_compatible
```

## 3) Rodar local sem Docker (opcional)

```bash
pip install -r requirements.txt
python src/train.py --training-source metadata
python src/evaluate.py
python src/predict.py caminho/para/musica.mp3
uvicorn src.api:app --reload
cd frontend && npm install && npm run dev
```

## 4) Features e abordagens

### Caminho A (principal no treino atual)
- Usa `features.csv` pré-computado + `tracks.csv`
- Gera tabela em `data/processed/fma_precomputed_features.csv`
- Não depende do `fma_small` para treinar

### Caminho B (opcional para comparação)
- Extração via `librosa` diretamente do áudio
- Mantido como alternativa/experimento

## 5) Treino e artefatos

Treino com `GridSearchCV` para:
- Decision Tree
- KNN
- Random Forest

Split:
- usa split oficial do FMA (`training`, `validation`, `test`) quando disponível
- fallback para split estratificado

Observação importante:
- O modelo continua aceitando MP3 novo.
- Na predição, o projeto escolhe automaticamente o extrator compatível com as colunas usadas no treino.

Artefatos gerados:
- `models/best_model.pkl`
- `models/scaler.pkl`
- `models/label_encoder.pkl`
- `models/feature_columns.pkl`
- `data/processed/model_comparison.csv`
- `data/processed/best_model_metrics.json`

## 6) Avaliação

Métricas:
- accuracy
- precision macro
- recall macro
- f1 macro
- matriz de confusão
- classification report

Arquivos de saída:
- `data/processed/classification_report.txt`
- `data/processed/confusion_matrix_best_model.png`
- `data/processed/model_comparison_metrics.png`
- `data/processed/random_forest_feature_importance.png` (quando aplicável)
- `data/processed/best_model_metrics_eval.json`

## 7) API FastAPI

Endpoints principais:
- `GET /api/health`
- `GET /api/model-info`
- `POST /api/predict` (upload `.mp3`/`.wav`)
- `GET /api/charts/genre-distribution`
- `GET /api/charts/model-comparison`
- `GET /api/charts/confusion-matrix`
- `GET /api/charts/feature-importance`
- `GET /api/charts/pca`

`POST /api/predict` retorna:
- gênero previsto
- probabilidades por gênero (se disponível)
- features relevantes do arquivo enviado
- comparação com média do gênero previsto e média global

## 8) Frontend React

Interface com:
- upload de `.mp3`/`.wav`
- gênero previsto
- ranking de probabilidades
- relação do arquivo enviado com features
- gráficos para artigo/apresentação

## 9) Streamlit (legado)

```bash
streamlit run src/app.py
```
