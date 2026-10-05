from __future__ import annotations

import difflib
import json
import math
import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseStage, MetricsCollector, StageContext
from sentence_transformers import SentenceTransformer
import sympy
from sympy.parsing.latex import parse_latex


@dataclass(frozen=True)
class VectorizationConfig:
    encoding: str = "utf-8"
    dense_model_path: str = "models/multilingual-e5-small"
    dense_model_repo: str = "intfloat/multilingual-e5-small"
    dense_model_auto_download: bool = True
    dense_device: str = "cpu"
    dense_max_seq_length: int = 512
    dense_passage_prefix: str = "passage: "
    dense_batch_size: int = 16
    sparse_method: str = "tfidf"
    token_source: str = "cl100k_base"
    restore_special_for_dense: bool = True
    use_normalized_tables: bool = True
    tables_stage_name: str = "normalization"
    tables_subdir: str = "tables"
    tables_dir: str = ""
    include_table_tokens_in_chunk_sparse: bool = True
    table_text_in_chunk_dense: bool = True

    max_table_chars: int = 4000

    formula_mode: str = "ast"
    table_mode: str = "header_type_rows"
    numbers_as_metadata: bool = True
    vector_dim_fallback: int = 384

    pca_variance: float = 0.90
    quantization: str = "none"

    vector_store: str = "qdrant"

    output_suffix: str = ".vectors"
    tables_output_suffix: str = ".tables.vectors"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.sparse_method not in ("tfidf", "bm25"):
            raise ValueError(f"unsupported sparse_method: {self.sparse_method}")

        if self.formula_mode not in ("single", "ast", "latex_aware"):
            raise ValueError(f"unsupported formula_mode: {self.formula_mode}")

        if self.table_mode not in ("header_type_rows", "markdown"):
            raise ValueError(f"unsupported table_mode: {self.table_mode}")

        if self.quantization not in ("none", "int8", "float16"):
            raise ValueError(f"unsupported quantization: {self.quantization}")

        if self.vector_store not in ("qdrant", "faiss", "milvus", "none"):
            raise ValueError(f"unsupported vector_store: {self.vector_store}")

        if self.max_table_chars <= 0:
            raise ValueError("max_table_chars must be positive")

        if self.dense_max_seq_length <= 0:
            raise ValueError("dense_max_seq_length must be positive")


@dataclass(frozen=True)
class NormalizedTable:
    source_file: str
    table_index: int
    header: List[str]
    columns: List[Dict[str, Any]]
    rows: List[Dict[str, Any]]
    markdown: str
    canonical_text: str
    dense_text: str
    sparse_tokens: List[str]
    formulas: List[Dict[str, Any]]


SPECIAL_RE = re.compile(
    r"\[(FORMULA|NUMUNIT|ABBREV|TABLE)\s*_(\d+)\]"
)

WORD_TOKEN_RE = re.compile(
    r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9,.\-]*",
    re.UNICODE,
)

FORMULA_RE = re.compile(
    r"\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|\$[^$\n]+?\$",
    re.S,
)

LATEX_TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+|[A-Za-z]+|\d+(?:[.,]\d+)?|[{}_^+-=*/()]"
)

NUMUNIT_PARSE_RE = re.compile(
    r"([-+]?\d+(?:[.,]\d+)?)\s*(.*)"
)

WHITESPACE_RE = re.compile(r"\s+")


class VectorizationStage(BaseStage):
    name = "vectorization"

    _dense_model = None
    _dense_model_failed = False
    _dense_model_name = None

    def __init__(self, config: Optional[VectorizationConfig] = None) -> None:
        super().__init__(config or VectorizationConfig())
        self._output_root: Optional[Path] = None

    def validate(self, ctx: StageContext) -> None:
        super().validate(ctx)
        self._output_root = ctx.output_dir

    def list_input_files(self, ctx: StageContext) -> List[Path]:
        return sorted(ctx.input_dir.glob("*.tokens.json"))

    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        metrics: MetricsCollector,
    ) -> Path:
        cfg: VectorizationConfig = self.config

        record_in = json.loads(input_path.read_text(encoding=cfg.encoding))
        source_name = record_in.get("source_file", input_path.name)
        chunks = record_in.get("chunks", [])

        file_metrics: Dict[str, Any] = {
            "chunks_in": len(chunks),
        }

        normalized_tables: List[NormalizedTable] = []

        if cfg.use_normalized_tables:
            normalized_tables = self._load_normalized_tables(
                source_name=source_name,
                input_path=input_path,
                file_metrics=file_metrics,
            )

        payloads: List[Dict[str, Any]] = []
        sparse_token_docs: List[List[str]] = []
        dense_texts: List[str] = []

        # Unique table texts to vectorize once.
        unique_table_texts: Dict[str, str] = {}

        for chunk in chunks:
            special = chunk.get("special", {})
            raw_tables = special.get("tables", {})

            table_dense_map: Dict[str, str] = {}
            chunk_table_entries: List[Dict[str, Any]] = []
            extra_sparse_tokens: List[str] = []

            for raw_idx, raw_table_entry in raw_tables.items():
                table_entry = self._prepare_table_entry(
                    raw_idx=str(raw_idx),
                    raw_table_entry=raw_table_entry,
                    normalized_tables=normalized_tables,
                    file_metrics=file_metrics,
                )

                unique_table_texts[table_entry["_vector_key"]] = table_entry["table_text"]

                if cfg.table_text_in_chunk_dense:
                    table_dense_map[str(raw_idx)] = table_entry["table_text"]

                if cfg.include_table_tokens_in_chunk_sparse:
                    extra_sparse_tokens.extend(table_entry.get("sparse_tokens", []))

                chunk_table_entries.append(table_entry)

            if extra_sparse_tokens:
                file_metrics["table_sparse_tokens_added"] = (
                    file_metrics.get("table_sparse_tokens_added", 0)
                    + len(extra_sparse_tokens)
                )

            sentence_text = " ".join(chunk.get("sentences", []))

            if cfg.restore_special_for_dense:
                dense_text = self._restore_special(
                    text=sentence_text,
                    special=special,
                    table_dense_map=table_dense_map,
                )
            else:
                dense_text = sentence_text

            cl100k_tokens = [
                token.strip()
                for token in chunk.get("tokens", [])
                if token.strip()
            ]

            sparse_tokens = cl100k_tokens + extra_sparse_tokens

            formulas = self._vectorize_formulas(
                special.get("formulas", {}),
                file_metrics,
            )

            numunits = self._index_numunits(
                special.get("numunit", {}),
                file_metrics,
            )

            abbreviations = self._registry_items(
                special.get("abbreviations", {})
            )

            payloads.append(
                {
                    "chunk_id": f"{source_name}::chunk::{chunk.get('index', 0)}",
                    "index": chunk.get("index", 0),
                    "token_source": cfg.token_source,
                    "token_count": chunk.get("token_count", len(cl100k_tokens)),
                    "cl100k_tokens": cl100k_tokens,
                    "sparse_tokens": sparse_tokens,
                    "dense_text": dense_text,
                    "abbreviations": abbreviations,
                    "formulas": formulas,
                    "tables": chunk_table_entries,
                    "numunits": numunits,
                }
            )

            sparse_token_docs.append(sparse_tokens)
            dense_texts.append(dense_text)

        sparse_vocab, sparse_vectors = self._build_sparse_vectors(sparse_token_docs)
        file_metrics["sparse_vocab_size"] = len(sparse_vocab)

        chunk_dense_vectors, chunk_dense_source = self._encode_or_hash_texts(
            texts=dense_texts,
            file_metrics=file_metrics,
            kind="chunk",
        )

        table_keys = list(unique_table_texts.keys())
        table_texts_to_encode = [unique_table_texts[key] for key in table_keys]

        table_vectors, table_vector_source = self._encode_or_hash_texts(
            texts=table_texts_to_encode,
            file_metrics=file_metrics,
            kind="table",
        )

        table_vector_map = dict(zip(table_keys, table_vectors))

        file_metrics["tables_vectorized"] = len(table_keys)

        if chunk_dense_source == "chunk_dense" or table_vector_source == "table_dense":
            file_metrics["dense_model_used"] = 1
        else:
            file_metrics["dense_model_fallback"] = 1

        # Assign table vectors and remove temporary keys.
        for payload in payloads:
            for table_entry in payload["tables"]:
                vector_key = table_entry.pop("_vector_key", None)
                table_entry["vector"] = table_vector_map.get(vector_key)
                table_entry["vector_source"] = table_vector_source

                # Sparse tokens are already included in chunk sparse vector.
                # Remove them from final artifact to reduce size.
                table_entry.pop("sparse_tokens", None)

        out_chunks: List[Dict[str, Any]] = []

        for idx, payload in enumerate(payloads):
            payload["sparse_method"] = cfg.sparse_method
            payload["sparse_vector"] = (
                sparse_vectors[idx]
                if idx < len(sparse_vectors)
                else {}
            )

            payload["dense_model"] = (
                VectorizationStage._dense_model_name
                if chunk_dense_source == "chunk_dense"
                else None
            )
            payload["dense_vector"] = (
                chunk_dense_vectors[idx]
                if idx < len(chunk_dense_vectors)
                else None
            )
            payload["dense_vector_source"] = chunk_dense_source

            if payload["dense_vector"] is not None:
                file_metrics["dense_vectors"] = (
                    file_metrics.get("dense_vectors", 0) + 1
                )

            if payload["sparse_vector"]:
                file_metrics["sparse_vectors"] = (
                    file_metrics.get("sparse_vectors", 0) + 1
                )

            out_chunks.append(payload)

        dense_vector_dim: Optional[int] = None

        for vector in chunk_dense_vectors + table_vectors:
            if vector is not None:
                dense_vector_dim = len(vector)
                break

        output_path = output_dir / f"{input_path.stem}{cfg.output_suffix}.json"

        output_record = {
            "source_file": source_name,
            "dense_model": VectorizationStage._dense_model_name,
            "dense_model_path": cfg.dense_model_path,
            "dense_model_repo": cfg.dense_model_repo,
            "dense_vector_dim": dense_vector_dim,
            "sparse_method": cfg.sparse_method,
            "token_source": cfg.token_source,
            "formula_mode": cfg.formula_mode,
            "table_mode": cfg.table_mode,
            "numbers_as_metadata": cfg.numbers_as_metadata,
            "vector_store": cfg.vector_store,
            "quantization": cfg.quantization,
            "pca_variance": cfg.pca_variance,
            "normalized_tables_loaded": len(normalized_tables),
            "table_vector_source": table_vector_source,
            "chunk_dense_vector_source": chunk_dense_source,
            "sparse_vocab_size": len(sparse_vocab),
            "chunks": out_chunks,
        }

        output_path.write_text(
            json.dumps(output_record, ensure_ascii=False, indent=2),
            encoding=cfg.encoding,
        )

        # Separate artifact with table vectors.
        tables_output_path = (
            output_dir
            / f"{input_path.stem}{cfg.tables_output_suffix}.json"
        )

        tables_record = {
            "source_file": source_name,
            "dense_model": VectorizationStage._dense_model_name,
            "table_vector_source": table_vector_source,
            "dense_vector_dim": dense_vector_dim,
            "tables": [
                {
                    "vector_key": key,
                    "table_text": unique_table_texts[key],
                    "vector": table_vector_map.get(key),
                }
                for key in table_keys
            ],
        }

        tables_output_path.write_text(
            json.dumps(tables_record, ensure_ascii=False, indent=2),
            encoding=cfg.encoding,
        )

        for key, value in file_metrics.items():
            if isinstance(value, int):
                metrics.inc(key, value)

        metrics.add_file_metrics(input_path.name, file_metrics)

        return output_path

    # ------------------------------------------------------------------
    # Normalized tables loading
    # ------------------------------------------------------------------

    def _load_normalized_tables(
        self,
        source_name: str,
        input_path: Path,
        file_metrics: Dict[str, Any],
    ) -> List[NormalizedTable]:
        cfg: VectorizationConfig = self.config

        tables_dir = self._get_tables_dir(input_path)

        if tables_dir is None or not tables_dir.is_dir():
            file_metrics["normalized_tables_dir_missing"] = 1
            return []

        source_stem = self._source_stem(source_name, input_path)
        tables_path = tables_dir / f"{source_stem}.tables.json"

        if not tables_path.exists():
            file_metrics["normalized_tables_file_missing"] = 1
            return []

        try:
            records = json.loads(
                tables_path.read_text(encoding=cfg.encoding)
            )
        except Exception:
            file_metrics["normalized_tables_file_invalid"] = 1
            return []

        if isinstance(records, dict):
            records = [records]

        if not isinstance(records, list):
            file_metrics["normalized_tables_file_invalid"] = 1
            return []

        normalized: List[NormalizedTable] = []

        for idx, record in enumerate(records):
            try:
                if not isinstance(record, dict):
                    continue

                if "table_index" not in record:
                    record = {**record, "table_index": idx}

                normalized.append(
                    self._build_normalized_table(record, source_name)
                )
            except Exception:
                file_metrics["normalized_tables_failed"] = (
                    file_metrics.get("normalized_tables_failed", 0) + 1
                )

        file_metrics["normalized_tables_loaded"] = len(normalized)
        return normalized

    def _get_tables_dir(self, input_path: Path) -> Optional[Path]:
        cfg: VectorizationConfig = self.config

        if cfg.tables_dir:
            return Path(cfg.tables_dir)

        if self._output_root is not None:
            return self._output_root / cfg.tables_stage_name / cfg.tables_subdir

        try:
            return (
                input_path.parents[1]
                / cfg.tables_stage_name
                / cfg.tables_subdir
            )
        except IndexError:
            return input_path.parent / cfg.tables_stage_name / cfg.tables_subdir

    def _source_stem(self, source_name: str, input_path: Path) -> str:
        source_name = str(source_name).strip()

        if source_name and source_name != input_path.name:
            return Path(source_name).stem

        name = input_path.name

        for suffix in (
            ".chunks.tokens.json",
            ".tokens.json",
            ".json",
        ):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        if name.endswith(".chunks"):
            name = name[: -len(".chunks")]

        return name

    def _build_normalized_table(
        self,
        record: Dict[str, Any],
        source_file: str,
    ) -> NormalizedTable:
        columns_raw = record.get("columns", [])
        rows_raw = record.get("rows", [])

        header: List[str] = []
        columns: List[Dict[str, Any]] = []

        if columns_raw:
            for idx, col in enumerate(columns_raw):
                name = str(col.get("name", f"col_{idx}"))
                header.append(name)
                columns.append(
                    {
                        "name": name,
                        "dtype": col.get("dtype", "str"),
                        "unit": col.get("unit"),
                    }
                )
        elif rows_raw and isinstance(rows_raw[0], dict):
            header = [str(key) for key in rows_raw[0].keys()]
            columns = [
                {"name": name, "dtype": "str", "unit": None}
                for name in header
            ]

        rows: List[Dict[str, Any]] = []

        for row in rows_raw:
            out_row: Dict[str, Any] = {}

            for name in header:
                if isinstance(row, dict):
                    value = row.get(name, "")
                else:
                    value = ""

                out_row[name] = "" if value is None else str(value)

            rows.append(out_row)

        markdown = self._rows_to_markdown(header, rows)
        canonical_text = self._canonical_text_from_cells(header, rows)

        table_index = int(record.get("table_index", 0))

        dense_text = self._table_dense_text(
            table_index=table_index,
            header=header,
            columns=columns,
            rows=rows,
        )

        sparse_tokens = self._table_sparse_tokens(
            header=header,
            columns=columns,
            rows=rows,
        )

        formulas = self._extract_table_formulas(header, rows)

        return NormalizedTable(
            source_file=source_file,
            table_index=table_index,
            header=header,
            columns=columns,
            rows=rows,
            markdown=markdown,
            canonical_text=canonical_text,
            dense_text=dense_text,
            sparse_tokens=sparse_tokens,
            formulas=formulas,
        )

    # ------------------------------------------------------------------
    # Table matching and preparation
    # ------------------------------------------------------------------

    def _prepare_table_entry(
        self,
        raw_idx: str,
        raw_table_entry: Any,
        normalized_tables: List[NormalizedTable],
        file_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        matched, score = self._match_raw_table(raw_table_entry, normalized_tables)

        if matched is not None:
            file_metrics["tables_matched"] = (
                file_metrics.get("tables_matched", 0) + 1
            )

            return {
                "id": raw_idx,
                "normalization_table_index": matched.table_index,
                "match_score": round(score, 4),
                "header": matched.header,
                "columns": matched.columns,
                "rows": matched.rows,
                "table_text": matched.dense_text,
                "sparse_tokens": matched.sparse_tokens,
                "formulas": matched.formulas,
                "_vector_key": f"normalized::{matched.table_index}",
            }

        file_metrics["tables_unmatched"] = (
            file_metrics.get("tables_unmatched", 0) + 1
        )

        header, rows = self._parse_markdown_cells(raw_table_entry)
        row_dicts = self._rows_to_dicts(header, rows)

        columns = [
            {"name": name, "dtype": "str", "unit": None}
            for name in header
        ]

        dense_text = self._table_dense_text(
            table_index=f"raw_{raw_idx}",
            header=header,
            columns=columns,
            rows=row_dicts,
        )

        sparse_tokens = self._table_sparse_tokens(
            header=header,
            columns=columns,
            rows=row_dicts,
        )

        formulas = self._extract_table_formulas(header, row_dicts)

        return {
            "id": raw_idx,
            "normalization_table_index": None,
            "match_score": round(score, 4),
            "header": header,
            "columns": columns,
            "rows": row_dicts,
            "table_text": dense_text,
            "sparse_tokens": sparse_tokens,
            "formulas": formulas,
            "_vector_key": f"raw::{raw_idx}",
        }

    def _match_raw_table(
        self,
        raw_table_entry: Dict[str, Any],
        normalized_tables: List[NormalizedTable],
    ) -> Tuple[Optional[NormalizedTable], float]:
        """
        Сопоставляет таблицу из токенизации с нормализованной таблицей по global_index.
        """
        if not normalized_tables:
            return None, 0.0
        
        if isinstance(raw_table_entry, dict) and "global_index" in raw_table_entry:
            global_index = raw_table_entry["global_index"]
            
            for table in normalized_tables:
                if table.table_index == global_index:
                    return table, 1.0
            
            return None, 0.0
        
        raw_table = raw_table_entry if isinstance(raw_table_entry, str) else str(raw_table_entry)

        best_table: Optional[NormalizedTable] = None
        best_score = 0.0

        for table in normalized_tables:
            score = difflib.SequenceMatcher(
                None,
                raw_table,
                table.canonical_text,
            ).ratio()

            if score > best_score:
                best_score = score
                best_table = table

        if best_table is not None and best_score >= 0.80:
            return best_table, best_score

        return None, best_score

    # ------------------------------------------------------------------
    # Table text / tokens / formulas
    # ------------------------------------------------------------------

    def _rows_to_markdown(
        self,
        header: List[str],
        rows: List[Dict[str, Any]],
    ) -> str:
        if not header:
            return ""

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]

        for row in rows:
            cells = [str(row.get(name, "")) for name in header]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _canonical_text_from_cells(
        self,
        header: List[str],
        rows: List[Dict[str, Any]],
    ) -> str:
        normalized_header = [self._normalize_cell(cell) for cell in header]

        normalized_rows: List[List[str]] = []

        for row in rows:
            normalized_rows.append(
                [
                    self._normalize_cell(row.get(name, ""))
                    for name in header
                ]
            )

        lines = ["|".join(normalized_header)]

        for row in normalized_rows:
            lines.append("|".join(row))

        return "\n".join(lines)

    def _normalize_cell(self, value: Any) -> str:
        return WHITESPACE_RE.sub(" ", str(value).strip())

    def _table_dense_text(
        self,
        table_index: Any,
        header: List[str],
        columns: List[Dict[str, Any]],
        rows: List[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []

        parts.append(f"Таблица {table_index}.")

        col_by_name = {
            str(col.get("name", "")): col
            for col in columns
        }

        column_descriptions: List[str] = []

        for name in header:
            col = col_by_name.get(name, {})
            dtype = col.get("dtype", "str")
            unit = col.get("unit")

            desc = f"{name}: {dtype}"

            if unit:
                desc += f", единица: {unit}"

            column_descriptions.append(desc)

        if column_descriptions:
            parts.append("Колонки: " + "; ".join(column_descriptions))

        for row_idx, row in enumerate(rows, start=1):
            row_parts: List[str] = []

            for name in header:
                value = row.get(name, "")
                row_parts.append(f"{name}: {value}")

            parts.append(f"Строка {row_idx}: " + "; ".join(row_parts))

        text = "\n".join(parts)
        return self._truncate_text(text, self.config.max_table_chars)

    def _table_sparse_tokens(
        self,
        header: List[str],
        columns: List[Dict[str, Any]],
        rows: List[Dict[str, Any]],
    ) -> List[str]:
        tokens: List[str] = []

        col_by_name = {
            str(col.get("name", "")): col
            for col in columns
        }

        for name in header:
            col = col_by_name.get(name, {})

            tokens.extend(self._text_tokens(str(name)))
            tokens.append(f"dtype:{col.get('dtype', 'str')}")

            if col.get("unit"):
                tokens.append(f"unit:{col['unit']}")

        for row in rows:
            for name in header:
                value = row.get(name, "")
                tokens.extend(self._text_tokens(str(value)))

        return tokens

    def _extract_table_formulas(
        self,
        header: List[str],
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        formulas: List[Dict[str, Any]] = []

        for row_idx, row in enumerate(rows, start=1):
            for name in header:
                value = str(row.get(name, ""))

                for match in FORMULA_RE.finditer(value):
                    latex = match.group(0)

                    formulas.append(
                        {
                            "row": row_idx,
                            "column": name,
                            "latex": latex,
                            "formula_tokens": self._formula_tokens(latex),
                        }
                    )

        return formulas

    def _parse_markdown_cells(
        self,
        table_text: str,
    ) -> Tuple[List[str], List[List[str]]]:
        lines = [
            line.strip()
            for line in str(table_text).splitlines()
            if line.strip()
        ]

        header: List[str] = []
        rows: List[List[str]] = []

        for line in lines:
            if self._is_table_separator(line):
                continue

            cells = self._split_table_row(line)

            if not header:
                header = cells
            else:
                rows.append(cells)

        return header, rows

    def _rows_to_dicts(
        self,
        header: List[str],
        rows: List[List[str]],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []

        if not header and rows:
            width = max(len(row) for row in rows)
            header = [f"col_{i}" for i in range(width)]

        for row in rows:
            row_dict: Dict[str, Any] = {}

            for idx, name in enumerate(header):
                row_dict[name] = row[idx] if idx < len(row) else ""

            result.append(row_dict)

        return result

    def _split_table_row(self, line: str) -> List[str]:
        stripped = line.strip()

        if stripped.startswith("|"):
            stripped = stripped[1:]

        if stripped.endswith("|"):
            stripped = stripped[:-1]

        parts = re.split(r"(?<!\\)\|", stripped)

        return [
            part.strip().replace("\\|", "|")
            for part in parts
        ]

    def _is_table_separator(self, line: str) -> bool:
        stripped = line.strip()

        if "|" not in stripped:
            return False

        if "-" not in stripped:
            return False

        cleaned = (
            stripped.replace("|", " ")
            .replace(":", "")
            .replace("-", "")
            .replace(" ", "")
        )

        return cleaned == ""

    # ------------------------------------------------------------------
    # Special entities restore
    # ------------------------------------------------------------------

    def _restore_special(
        self,
        text: str,
        special: Dict[str, Any],
        table_dense_map: Optional[Dict[str, str]] = None,
    ) -> str:
        kind_to_registry = {
            "FORMULA": "formulas",
            "NUMUNIT": "numunit",
            "ABBREV": "abbreviations",
            "TABLE": "tables",
        }

        def repl(match: re.Match) -> str:
            kind = match.group(1)
            idx = match.group(2)

            if kind == "TABLE" and table_dense_map and idx in table_dense_map:
                return table_dense_map[idx]

            registry_name = kind_to_registry.get(kind)

            if registry_name is None:
                return match.group(0)

            entry = special.get(registry_name, {}).get(idx)

            if entry is None:
                return match.group(0)

            if isinstance(entry, dict):
                return str(entry.get("content", "")).strip()

            return str(entry).strip()

        return SPECIAL_RE.sub(repl, text)

    def _registry_items(self, registry: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"id": str(key), "text": str(value).strip()}
            for key, value in registry.items()
        ]

    # ------------------------------------------------------------------
    # Formulas / numunits
    # ------------------------------------------------------------------

    def _vectorize_formulas(
        self,
        registry: Dict[str, Any],
        file_metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []

        for idx, entry in registry.items():
            if isinstance(entry, dict):
                latex = str(entry.get("content", "")).strip()
            else:
                latex = str(entry).strip()

            formula_tokens = self._formula_tokens(latex)
            vector = self._hash_vector(formula_tokens)

            result.append(
                {
                    "id": str(idx),
                    "latex": latex,
                    "formula_tokens": formula_tokens,
                    "vector": vector,
                }
            )

            file_metrics["formulas_vectorized"] = (
                file_metrics.get("formulas_vectorized", 0) + 1
            )

        return result

    def _formula_tokens(self, latex: str) -> List[str]:
        cfg: VectorizationConfig = self.config

        if cfg.formula_mode == "single":
            return [latex]

        if cfg.formula_mode == "ast":
            pieces = self._ast_pieces(latex)

            if pieces:
                return pieces

        return LATEX_TOKEN_RE.findall(self._strip_delimiters(latex))

    def _ast_pieces(self, latex: str) -> Optional[List[str]]:
        if parse_latex is None or sympy is None:
            return None

        try:
            expr = parse_latex(self._strip_delimiters(latex))
        except Exception:
            return None

        pieces: List[str] = []

        for node in sympy.preorder_traversal(expr):
            if node.is_Number:
                pieces.append(str(node))
            elif node.is_Symbol:
                pieces.append(str(node))
            else:
                pieces.append(type(node).__name__)

        return pieces or None

    def _strip_delimiters(self, latex: str) -> str:
        if latex.startswith("$$") and latex.endswith("$$"):
            return latex[2:-2]

        if latex.startswith("\\[") and latex.endswith("\\]"):
            return latex[2:-2]

        if latex.startswith("\\(") and latex.endswith("\\)"):
            return latex[2:-2]

        if latex.startswith("$") and latex.endswith("$") and len(latex) > 1:
            return latex[1:-1]

        return latex

    def _index_numunits(
        self,
        registry: Dict[str, Any],
        file_metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        cfg: VectorizationConfig = self.config
        result: List[Dict[str, Any]] = []

        if not cfg.numbers_as_metadata:
            return result

        for idx, content in registry.items():
            raw = str(content).strip()
            match = NUMUNIT_PARSE_RE.match(raw)

            value: Optional[float] = None
            unit: Optional[str] = None

            if match:
                try:
                    value = float(match.group(1).replace(",", "."))
                except Exception:
                    value = None

                unit = match.group(2).strip() or None

            result.append(
                {
                    "id": str(idx),
                    "raw": raw,
                    "value": value,
                    "unit": unit,
                }
            )

            file_metrics["numunits_indexed"] = (
                file_metrics.get("numunits_indexed", 0) + 1
            )

        return result

    # ------------------------------------------------------------------
    # Sparse vectors
    # ------------------------------------------------------------------

    def _build_sparse_vectors(
        self,
        token_docs: List[List[str]],
    ) -> Tuple[Dict[str, int], List[Dict[int, float]]]:
        cfg: VectorizationConfig = self.config

        vocab: Dict[str, int] = {}
        df: Dict[str, int] = {}
        doc_lengths: List[int] = []

        for doc in token_docs:
            doc_lengths.append(len(doc))
            seen = set()

            for token in doc:
                if token not in vocab:
                    vocab[token] = len(vocab)

                if token not in seen:
                    df[token] = df.get(token, 0) + 1
                    seen.add(token)

        n = len(token_docs)

        if n == 0:
            return vocab, []

        avgdl = sum(doc_lengths) / n if n else 1.0
        vectors: List[Dict[int, float]] = []

        for doc in token_docs:
            tf: Dict[str, int] = {}

            for token in doc:
                tf[token] = tf.get(token, 0) + 1

            vector: Dict[int, float] = {}

            for token, count in tf.items():
                token_id = vocab[token]

                if cfg.sparse_method == "bm25":
                    idf = math.log(
                        (n - df[token] + 0.5) / (df[token] + 0.5) + 1.0
                    )

                    k1 = 1.5
                    b = 0.75

                    denominator = count + k1 * (
                        1.0 - b + b * (len(doc) / avgdl)
                    )

                    score = idf * (count * (k1 + 1.0)) / denominator

                else:
                    idf = math.log((1 + n) / (1 + df[token])) + 1.0
                    score = (count / len(doc)) * idf if doc else 0.0

                vector[token_id] = round(score, 6)

            vectors.append(vector)

        return vocab, vectors

    # ------------------------------------------------------------------
    # Dense vectors / fallback
    # ------------------------------------------------------------------

    def _encode_or_hash_texts(
        self,
        texts: List[str],
        file_metrics: Dict[str, Any],
        kind: str,
    ) -> Tuple[List[List[float]], str]:
        if not texts:
            return [], "none"

        dense_vectors = self._encode_dense(
            texts=texts,
            file_metrics=file_metrics,
            kind=kind,
        )

        if (
            dense_vectors
            and len(dense_vectors) == len(texts)
            and all(vector is not None for vector in dense_vectors)
        ):
            return dense_vectors, f"{kind}_dense"

        vectors: List[List[float]] = []

        for text in texts:
            tokens = self._text_tokens(text)
            vectors.append(self._hash_vector(tokens))

        file_metrics[f"{kind}_hash_fallback"] = (
            file_metrics.get(f"{kind}_hash_fallback", 0) + len(texts)
        )

        return vectors, f"{kind}_hash_fallback"

    def _encode_dense(
        self,
        texts: List[str],
        file_metrics: Dict[str, Any],
        kind: str,
    ) -> List[Optional[List[float]]]:
        if not texts:
            return []

        model = self._get_dense_model()

        if model is None:
            file_metrics["dense_model_unavailable"] = (
                file_metrics.get("dense_model_unavailable", 0) + 1
            )
            return [None for _ in texts]

        vectors: List[Optional[List[float]]] = []

        try:
            batch_size = self.config.dense_batch_size
            prefix = self.config.dense_passage_prefix or ""

            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]

                if prefix:
                    batch = [prefix + (text or "") for text in batch]

                embeddings = model.encode(
                    batch,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for row in embeddings:
                    vectors.append([float(x) for x in row])

            if len(vectors) != len(texts):
                raise ValueError("dense model returned unexpected vector count")

            file_metrics[f"{kind}_dense_encoded"] = (
                file_metrics.get(f"{kind}_dense_encoded", 0) + len(texts)
            )

        except Exception:
            file_metrics["dense_model_unavailable"] = (
                file_metrics.get("dense_model_unavailable", 0) + 1
            )
            vectors = [None for _ in texts]

        return vectors

    def _get_dense_model(self):
        cfg: VectorizationConfig = self.config

        if VectorizationStage._dense_model is not None:
            return VectorizationStage._dense_model

        if VectorizationStage._dense_model_failed:
            return None

        if SentenceTransformer is None:
            VectorizationStage._dense_model_failed = True
            return None

        local_path: Optional[Path] = None

        if cfg.dense_model_path:
            local_path = Path(cfg.dense_model_path)

        if (
            local_path is not None
            and not local_path.exists()
            and cfg.dense_model_auto_download
            and cfg.dense_model_repo
        ):
            self._download_model(cfg.dense_model_repo, local_path)

        candidates: List[str] = []

        if local_path is not None and local_path.exists():
            candidates.append(str(local_path))

        if cfg.dense_model_repo:
            candidates.append(cfg.dense_model_repo)

        if local_path is not None and str(local_path) not in candidates:
            candidates.append(str(local_path))

        for candidate in candidates:
            try:
                model = SentenceTransformer(
                    candidate,
                    device=cfg.dense_device,
                )

                if cfg.dense_max_seq_length > 0 and hasattr(model, "max_seq_length"):
                    model.max_seq_length = cfg.dense_max_seq_length

                VectorizationStage._dense_model = model
                VectorizationStage._dense_model_name = candidate
                return model
            except Exception:
                continue

        VectorizationStage._dense_model_failed = True
        return None

    def _download_model(self, repo_id: str, local_dir: Path) -> None:
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            return

        try:
            local_dir.mkdir(parents=True, exist_ok=True)

            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                )
            except TypeError:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                )
        except Exception:
            return

    def _hash_vector(self, tokens: List[str]) -> List[float]:
        cfg: VectorizationConfig = self.config
        dim = cfg.vector_dim_fallback

        if not tokens:
            return [0.0] * dim

        vector = [0.0] * dim

        for token in tokens:
            token_bytes = str(token).encode(cfg.encoding)
            idx = zlib.crc32(token_bytes) % dim
            vector[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vector)) or 1.0

        return [round(x / norm, 6) for x in vector]

    def _text_tokens(self, text: str) -> List[str]:
        tokens: List[str] = []

        if not text:
            return tokens

        last = 0

        for match in FORMULA_RE.finditer(text):
            if match.start() > last:
                tokens.extend(
                    WORD_TOKEN_RE.findall(text[last : match.start()])
                )

            tokens.append(match.group(0))
            last = match.end()

        if last < len(text):
            tokens.extend(WORD_TOKEN_RE.findall(text[last:]))

        return tokens

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text

        return text[:max_chars] + " ..."

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def finalize_metrics(self, metrics: MetricsCollector) -> None:
        quality_names = [
            "hit_rate_at_10",
            "mrr",
            "ndcg_at_10",
            "cosine_similarity",
            "mteb",
            "beir",
            "ru_mteb",
            "table_structured_coverage",
        ]

        for name in quality_names:
            metrics.set_quality(name, None)

        metrics.meta["quality_note"] = (
            "Качественные метрики векторизации требуют эталонного набора "
            "запросов и релевантных чанков. Для таблиц отдельно проверяется "
            "покрытие структурированными таблицами из normalization."
        )

        metrics.meta["expected_quality_metrics"] = quality_names

        metrics.meta["thresholds"] = {
            "hit_rate_at_10": "> 0.9",
            "mrr": "monitor",
            "ndcg_at_10": "monitor",
            "cosine_similarity": "monitor",
            "mteb": "monitor",
            "beir": "monitor",
            "ru_mteb": "monitor",
            "table_structured_coverage": "monitor",
        }

        metrics.meta["libraries"] = [
            "sklearn",
            "sentence-transformers",
            "E5",
            "GTE",
            "SBERT",
            "ColBERT",
            "CLIP",
            "ImageBind",
            "SymPy",
            "pandas",
            "FAISS",
            "Qdrant",
            "Milvus",
            "Elasticsearch",
        ]

        metrics.meta["note"] = (
            "cl100k_base используется для разреженного поиска и контроля "
            "длины блоков. Таблицы векторизуются через структурированные "
            "записи из normalization/tables: заголовки, типы, единицы, "
            "строки и формулы. Для плотных эмбеддингов используется "
            "локальная модель, если задан путь в dense_model_path."
        )