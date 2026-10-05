# Задание: построение графа по предобработанному тексту

Пайплайн построения графа знаний из русскоязычных научно-технических текстов по металлургии и производству стальных труб. На вход — Markdown-файлы, на выходе — граф в формате GraphML.

## Этапы

Каждый этап читает результат предыдущего и пишет артефакты и `metrics.json` в `output/<stage>/`.

| # | Этап | Что делает |
|---|------|-----------|
| 1 | `clearing` | Чистка Markdown: изображения, ссылки, разметка; HTML-таблицы → Markdown; защита формул и таблиц |
| 2 | `normalization` | Нормализация единиц, дат, чисел, формул (SymPy), таблиц; лемматизация (pymorphy3); замена аббревиатур |
| 3 | `chunking` | Разбиение на чанки по 800 токенов (`cl100k_base`) с перекрытием 15 % |
| 4 | `tokenization` | Гибридная токенизация, формулы как отдельные токены |
| 5 | `vectorization` | Dense-векторы (`intfloat/multilingual-e5-small`) + sparse TF-IDF для чанков и таблиц |
| 6 | `graph_building` | Извлечение сущностей и связей LLM (`Qwen/Qwen3.5-9B`) по онтологии, сборка графа (NetworkX) |

Итог: `output/graph_building/knowledge_graph.graphml`.

## Структура

```
main.py            # точка входа, конфигурация этапов
constants.py       # ключи для метрик и результатов этапов
pipeline/          # реализации этапов (base, clearing, normalization, chunking, ...)
configs/
  config.json      # параметры построения графа
  ontology.json    # типы сущностей и связей
  llm.json         # параметры LLM
  extract.txt      # промпт для извлечения графа
input/             # входные .md-файлы (создать вручную)
models/            # локальные модели (скачиваются автоматически)
output/            # результаты (создаётся автоматически)
```

## Установка

Требуется Python ≥ 3.10.

```bash
# вариант с uv
uv sync

# вариант с pip
python -m venv .venv && source .venv/bin/activate
pip install .
```

> Проект запускается из исходников: `configs/` и `models/` ищутся рядом с `main.py`.

## Запуск

```bash
mkdir -p input
cp your_papers/*.md input/
python main.py          # или: uv run python main.py
```

При первом запуске нужен интернет: скачиваются `multilingual-e5-small`, `Qwen3.5-9B` (в `models/`), а также кодировка `tiktoken` и, при необходимости, стоп-слова NLTK (`python -m nltk.downloader stopwords`).

## Настройка

- Параметры этапов (чистка, чанки, векторизация, словарь аббревиатур и терминов) задаются прямо в `main.py`.
- Онтология графа — `configs/ontology.json`, промпт — `configs/extract.txt`, ограничения на узлы и рёбра — `configs/config.json`.
- LLM (путь, устройство, длина генерации) — `configs/llm.json`. Для GPU поставьте `"llm_device": "cuda"`.

## Замечания

- На CPU модель загружается в `float32`: Qwen3.5-9B потребует порядка 36 ГБ ОЗУ и работает медленно. Для ускорения используйте GPU.
- Параметр `vector_store="qdrant"` в `VectorizationConfig` пока только метка в метаданных: клиент Qdrant не подключается, векторы сохраняются в JSON.
- Все входные файлы должны иметь расширение `.md`.
