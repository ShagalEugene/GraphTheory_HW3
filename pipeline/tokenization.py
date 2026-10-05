from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
import sympy
from sympy.parsing.latex import parse_latex

from .base import BaseStage, MetricsCollector, StageContext
from .protect import DISPLAY_MATH_RE, INLINE_MATH_RE, NUMUNIT_RE


@dataclass(frozen=True)
class TokenizationConfig:
    encoding: str = "utf-8"
    mode: str = "hybrid"
    token_encoding: str = "cl100k_base"

    protect_formulas: bool = True
    formula_token_mode: str = "single"

    protect_numunit: bool = True
    extract_tables: bool = True
    abbreviations_as_tokens: bool = True

    terms: Tuple[str, ...] = ()

    output_suffix: str = ".tokens"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.mode not in ("word", "subword", "hybrid"):
            raise ValueError(f"unsupported tokenization mode: {self.mode}")

        if self.formula_token_mode not in ("single", "ast", "latex_aware"):
            raise ValueError(
                f"unsupported formula token mode: {self.formula_token_mode}"
            )


SENTENCE_SPLIT_RE = re.compile(r"(?<![A-ZА-ЯЁ]\.)(?<=[.!?])\s+")

SPECIAL_TOKEN_RE = re.compile(r"\[(?:FORMULA|NUMUNIT|ABBREV|TABLE)_\d+\]")

WORD_TOKEN_RE = re.compile(
    r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]*|\d+(?:[.,]\d+)?"
)

LATEX_TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+|[A-Za-z]+|\d+(?:[.,]\d+)?|[{}_^+\-=*/()]"
)

WHITESPACE_RE = re.compile(r"\s+")


class TokenizationStage(BaseStage):
    name = "tokenization"

    _encoder = None
    _encoder_failed = False

    def __init__(self, config: Optional[TokenizationConfig] = None) -> None:
        super().__init__(config or TokenizationConfig())
        self._term_patterns: Optional[List[re.Pattern]] = None

    def list_input_files(self, ctx: StageContext) -> List[Path]:
        return sorted(ctx.input_dir.glob("*.chunks.json"))

    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        metrics: MetricsCollector,
    ) -> Path:
        cfg: TokenizationConfig = self.config

        record_in = json.loads(input_path.read_text(encoding=cfg.encoding))
        source_name = record_in.get("source_file", input_path.name)
        chunks_in = record_in.get("chunks", [])

        file_metrics: Dict[str, Any] = {"chunks_in": len(chunks_in)}

        vocab: Dict[str, Any] = {
            "terms": list(cfg.terms),
            "formulas": {},
            "numunit": {},
            "abbreviations": {},
            "tables": {},
        }

        out_chunks: List[Dict[str, Any]] = []
        original_texts: List[str] = []

        global_table_index = 0

        for chunk in chunks_in:
            chunk_out, chunk_metrics, registries = self._tokenize_chunk(
                chunk, cfg, global_table_index
            )

            for key, value in chunk_metrics.items():
                file_metrics[key] = file_metrics.get(key, 0) + value

            out_chunks.append(chunk_out)
            original_texts.append(chunk.get("text", ""))

            chunk_key = str(chunk_out["index"])

            if registries["FORMULA"]:
                vocab["formulas"][chunk_key] = registries["FORMULA"]
            if registries["NUMUNIT"]:
                vocab["numunit"][chunk_key] = registries["NUMUNIT"]
            if registries["ABBREV"]:
                vocab["abbreviations"][chunk_key] = registries["ABBREV"]
            if registries["TABLE"]:
                vocab["tables"][chunk_key] = registries["TABLE"]
                global_table_index += len(registries["TABLE"])

        full_text = "\n".join(original_texts)

        self._compute_oov(full_text, cfg, file_metrics)

        file_metrics["tokens_total"] = sum(
            chunk["token_count"] for chunk in out_chunks
        )
        file_metrics["sentences_total"] = sum(
            len(chunk["sentences"]) for chunk in out_chunks
        )

        tokens_path = output_dir / f"{input_path.stem}{cfg.output_suffix}.json"
        tokens_path.write_text(
            json.dumps(
                {
                    "source_file": source_name,
                    "mode": cfg.mode,
                    "formula_token_mode": cfg.formula_token_mode,
                    "chunks": out_chunks,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding=cfg.encoding,
        )

        vocab_path = output_dir / f"{input_path.stem}.vocab.json"
        vocab_path.write_text(
            json.dumps(vocab, ensure_ascii=False, indent=2),
            encoding=cfg.encoding,
        )

        for key, value in file_metrics.items():
            if isinstance(value, int):
                metrics.inc(key, value)

        metrics.add_file_metrics(input_path.name, file_metrics)

        return tokens_path

    def _get_term_patterns(self) -> List[re.Pattern]:
        if self._term_patterns is None:
            self._term_patterns = [
                re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])")
                for term in sorted(set(self.config.terms), key=len, reverse=True)
            ]

        return self._term_patterns

    def _select_term_matches(
        self,
        text: str,
    ) -> List[Tuple[int, int, str]]:
        matches: List[Tuple[int, int, str, int]] = []

        for pattern_index, pattern in enumerate(self._get_term_patterns()):
            for match in pattern.finditer(text):
                matches.append(
                    (
                        match.start(),
                        match.end(),
                        match.group(0),
                        pattern_index,
                    )
                )

        matches.sort(
            key=lambda item: (
                -(item[1] - item[0]),
                item[0],
                item[3],
            )
        )

        selected: List[Tuple[int, int, str]] = []
        occupied: List[Tuple[int, int]] = []

        for start, end, value, _ in matches:
            overlaps = False

            for occ_start, occ_end in occupied:
                if not (end <= occ_start or start >= occ_end):
                    overlaps = True
                    break

            if overlaps:
                continue

            occupied.append((start, end))
            selected.append((start, end, value))

        selected.sort(key=lambda item: item[0])
        return selected

    def _protect_selected_terms(
        self,
        text: str,
        matches: List[Tuple[int, int, str]],
        registries: Dict[str, Dict[str, Any]],
        metrics_local: Dict[str, int],
    ) -> str:
        registry = registries["ABBREV"]
        replacements: List[Tuple[int, int, str]] = []

        for start, end, content in matches:
            idx = len(registry)
            registry[str(idx)] = content

            metrics_local["abbreviations_protected"] = (
                metrics_local.get("abbreviations_protected", 0) + 1
            )

            replacements.append((start, end, f"[ABBREV_{idx}]"))

        for start, end, marker in reversed(replacements):
            text = text[:start] + marker + text[end:]

        return text

    def _tokenize_chunk(
        self,
        chunk: Dict[str, Any],
        cfg: TokenizationConfig,
        global_table_index: int = 0,
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Dict[str, Any]]]:
        text = chunk.get("text", "")
        index = chunk.get("index", 0)

        metrics_local: Dict[str, int] = {}
        registries: Dict[str, Dict[str, Any]] = {
            "FORMULA": {},
            "NUMUNIT": {},
            "ABBREV": {},
            "TABLE": {},
        }

        if cfg.extract_tables:
            text = self._extract_tables(
                text, registries["TABLE"], metrics_local, global_table_index
            )

        metrics_local["formulas_found"] = metrics_local.get("formulas_found", 0) + len(
            DISPLAY_MATH_RE.findall(text)
        ) + len(INLINE_MATH_RE.findall(text))

        if cfg.protect_formulas:
            text = self._protect_pattern(
                text, DISPLAY_MATH_RE, "FORMULA", registries, metrics_local, cfg
            )
            text = self._protect_pattern(
                text, INLINE_MATH_RE, "FORMULA", registries, metrics_local, cfg
            )

        metrics_local["numunit_found"] = metrics_local.get("numunit_found", 0) + len(
            NUMUNIT_RE.findall(text)
        )

        if cfg.protect_numunit:
            text = self._protect_pattern(
                text, NUMUNIT_RE, "NUMUNIT", registries, metrics_local, cfg
            )

        terms_found = 0

        for pattern in self._get_term_patterns():
            terms_found += len(pattern.findall(text))

        metrics_local["terms_found"] = metrics_local.get("terms_found", 0) + terms_found
        metrics_local["abbreviations_found"] = (
            metrics_local.get("abbreviations_found", 0) + terms_found
        )

        if cfg.abbreviations_as_tokens:
            for pattern in self._get_term_patterns():
                text = self._protect_pattern(
                    text, pattern, "ABBREV", registries, metrics_local, cfg
                )

        sentences = [
            sentence.strip()
            for sentence in SENTENCE_SPLIT_RE.split(text)
            if sentence.strip()
        ]

        tokens: List[str] = []
        recon_sentences: List[str] = []

        for sentence in sentences:
            sentence_tokens, sentence_recon = self._tokenize_segment(
                sentence, cfg, registries
            )
            tokens.extend(sentence_tokens)
            recon_sentences.append(sentence_recon)

        if cfg.mode != "word":
            reconstructed = " ".join(recon_sentences)
            expected = WHITESPACE_RE.sub(" ", chunk.get("text", "")).strip()
            actual = WHITESPACE_RE.sub(" ", reconstructed).strip()

            if actual == expected:
                metrics_local["roundtrip_ok"] = (
                    metrics_local.get("roundtrip_ok", 0) + 1
                )
            else:
                metrics_local["roundtrip_failed"] = (
                    metrics_local.get("roundtrip_failed", 0) + 1
                )

        chunk_out = {
            "index": index,
            "tokens": tokens,
            "token_count": len(tokens),
            "sentences": sentences,
            "special": {
                "formulas": registries["FORMULA"],
                "numunit": registries["NUMUNIT"],
                "abbreviations": registries["ABBREV"],
                "tables": registries["TABLE"],
            },
        }

        return chunk_out, metrics_local, registries

    def _protect_pattern(
        self,
        text: str,
        pattern: re.Pattern,
        kind: str,
        registries: Dict[str, Dict[str, Any]],
        metrics_local: Dict[str, int],
        cfg: TokenizationConfig,
    ) -> str:
        registry = registries[kind]

        def repl(match: re.Match) -> str:
            idx = len(registry)
            content = match.group(0)

            if kind == "FORMULA":
                registry[str(idx)] = {
                    "content": content,
                    "tokens": self._formula_tokens(content, cfg, metrics_local),
                }
                metrics_local["formulas_protected"] = (
                    metrics_local.get("formulas_protected", 0) + 1
                )
            elif kind == "NUMUNIT":
                registry[str(idx)] = content
                metrics_local["numunit_merged"] = (
                    metrics_local.get("numunit_merged", 0) + 1
                )
            else:
                registry[str(idx)] = content
                metrics_local["abbreviations_protected"] = (
                    metrics_local.get("abbreviations_protected", 0) + 1
                )

            return f"[{kind}_{idx}]"

        return pattern.sub(repl, text)

    def _formula_tokens(
        self,
        latex: str,
        cfg: TokenizationConfig,
        metrics_local: Dict[str, int],
    ) -> Optional[List[str]]:
        if cfg.formula_token_mode == "single":
            metrics_local["formulas_single_token"] = (
                metrics_local.get("formulas_single_token", 0) + 1
            )
            return None

        if cfg.formula_token_mode == "ast":
            pieces = self._ast_pieces(latex)

            if pieces is None:
                metrics_local["formulas_ast_fallback"] = (
                    metrics_local.get("formulas_ast_fallback", 0) + 1
                )
                return None

            return pieces

        if cfg.formula_token_mode == "latex_aware":
            return LATEX_TOKEN_RE.findall(self._strip_delimiters(latex))

        return None

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
        if latex.startswith("$$"):
            return latex[2:-2]
        if latex.startswith("\\["):
            return latex[2:-2]
        if latex.startswith("\\("):
            return latex[2:-2]
        return latex[1:-1]

    def _extract_tables(
        self,
        text: str,
        registry: Dict[str, Any],
        metrics_local: Dict[str, int],
        global_table_index: int = 0,
    ) -> str:
        lines = text.split("\n")
        out: List[str] = []
        block: List[str] = []

        def flush() -> None:
            if len(block) >= 2:
                idx = len(registry)
                registry[str(idx)] = {
                    "content": "\n".join(block),
                    "global_index": global_table_index + idx,
                }
                out.append(f"[TABLE_{idx}]")
                metrics_local["tables_extracted"] = (
                    metrics_local.get("tables_extracted", 0) + 1
                )
            elif block:
                out.extend(block)
            block.clear()

        for line in lines:
            stripped = line.strip()

            if stripped.count("|") >= 2:
                prefix = ""
                body = stripped

                if not stripped.startswith("|"):
                    first = stripped.index("|")
                    prefix = stripped[:first].strip()
                    body = stripped[first:]

                suffix = ""

                if not body.endswith("|"):
                    last = body.rindex("|")
                    suffix = body[last + 1:].strip()
                    body = body[: last + 1]

                if prefix:
                    if block:
                        flush()
                    out.append(prefix)

                block.append(body)

                if suffix:
                    flush()
                    out.append(suffix)

                continue

            if block:
                flush()

            if stripped:
                out.append(stripped)

        if block:
            flush()

        return "\n".join(out)

    def _tokenize_segment(
        self,
        segment: str,
        cfg: TokenizationConfig,
        registries: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], str]:
        tokens: List[str] = []
        recon_parts: List[str] = []
        pos = 0

        for match in SPECIAL_TOKEN_RE.finditer(segment):
            if match.start() > pos:
                raw = segment[pos:match.start()]
                pieces, ids = self._tokenize_plain(raw, cfg)
                tokens.extend(pieces)
                recon_parts.append(self._decode_ids(ids) if ids else raw)

            tokens.extend(self._expand_special(match.group(0), registries))
            recon_parts.append(self._special_text(match.group(0), registries))
            pos = match.end()

        if pos < len(segment):
            raw = segment[pos:]
            pieces, ids = self._tokenize_plain(raw, cfg)
            tokens.extend(pieces)
            recon_parts.append(self._decode_ids(ids) if ids else raw)

        return tokens, "".join(recon_parts)

    def _special_text(
        self,
        token_text: str,
        registries: Dict[str, Dict[str, Any]],
    ) -> str:
        kind, idx = token_text[1:-1].rsplit("_", 1)
        entry = registries.get(kind, {}).get(idx)

        if isinstance(entry, dict):
            return str(entry.get("content", ""))

        if isinstance(entry, str):
            return entry

        return ""

    def _decode_ids(self, ids: List[int]) -> str:
        encoder = self._get_encoder()

        if encoder is None or not ids:
            return ""

        return encoder.decode(ids)

    def _expand_special(
        self,
        token_text: str,
        registries: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        kind, idx = token_text[1:-1].rsplit("_", 1)
        registry = registries.get(kind, {})
        entry = registry.get(idx)

        if entry is None:
            return [token_text]

        if kind == "FORMULA" and isinstance(entry, dict) and entry.get("tokens"):
            return list(entry["tokens"])

        return [token_text]

    def _tokenize_plain(
        self,
        text: str,
        cfg: TokenizationConfig,
    ) -> Tuple[List[str], List[int]]:
        if not text.strip():
            return [], []

        if cfg.mode == "word":
            return WORD_TOKEN_RE.findall(text), []

        encoder = self._get_encoder()

        if encoder is None:
            return WORD_TOKEN_RE.findall(text), []

        ids = encoder.encode(text, disallowed_special=())
        pieces: List[str] = []

        for token_id in ids:
            try:
                pieces.append(
                    encoder.decode_single_token_bytes(token_id).decode(
                        "utf-8", errors="replace"
                    )
                )
            except Exception:
                pieces.append(str(token_id))

        return pieces, ids

    def _compute_oov(
        self,
        text: str,
        cfg: TokenizationConfig,
        file_metrics: Dict[str, Any],
    ) -> None:
        if cfg.mode == "word":
            file_metrics["oov_unique_words"] = 0
            file_metrics["oov_fragmented_words"] = 0
            file_metrics["oov_byte_fallback_words"] = 0
            return

        encoder = self._get_encoder()

        if encoder is None:
            file_metrics["oov_unique_words"] = 0
            file_metrics["oov_fragmented_words"] = 0
            file_metrics["oov_byte_fallback_words"] = 0
            return

        terms = set(cfg.terms)
        words = {
            word
            for word in WORD_TOKEN_RE.findall(text)
            if len(word) >= 3 and word not in terms
        }

        fragmented = 0
        byte_fallback = 0

        for word in words:
            ids = encoder.encode(word, disallowed_special=())

            if len(ids) > 1:
                fragmented += 1

            dirty = False

            for token_id in ids:
                try:
                    encoder.decode_single_token_bytes(token_id).decode("utf-8")
                except Exception:
                    dirty = True
                    break

            if dirty:
                byte_fallback += 1

        file_metrics["oov_unique_words"] = len(words)
        file_metrics["oov_fragmented_words"] = fragmented
        file_metrics["oov_byte_fallback_words"] = byte_fallback

    def _get_encoder(self):
        if TokenizationStage._encoder is None and not TokenizationStage._encoder_failed:
            try:
                TokenizationStage._encoder = tiktoken.get_encoding(
                    self.config.token_encoding
                )
            except Exception:
                TokenizationStage._encoder_failed = True

        return TokenizationStage._encoder

    def finalize_metrics(self, metrics: MetricsCollector) -> None:
        counters = metrics.counters

        def ratio(num: int, den: int) -> Optional[float]:
            return round(num / den, 4) if den else None

        metrics.set_quality(
            "terms_preserved_ratio",
            ratio(
                counters.get("abbreviations_protected", 0),
                counters.get("terms_found", 0),
            ),
        )
        metrics.set_quality(
            "formulas_correct_ratio",
            ratio(
                counters.get("formulas_single_token", 0),
                counters.get("formulas_protected", 0),
            )
            if self.config.formula_token_mode == "single"
            else ratio(
                counters.get("formulas_protected", 0)
                - counters.get("formulas_ast_fallback", 0),
                counters.get("formulas_protected", 0),
            ),
        )
        metrics.set_quality(
            "numunit_merged_ratio",
            ratio(counters.get("numunit_merged", 0), counters.get("numunit_found", 0)),
        )
        metrics.set_quality(
            "abbreviations_correct_ratio",
            ratio(
                counters.get("abbreviations_protected", 0),
                counters.get("abbreviations_found", 0),
            ),
        )
        metrics.set_quality(
            "oov_rate",
            ratio(
                counters.get("oov_byte_fallback_words", 0),
                counters.get("oov_unique_words", 0),
            ),
        )
        metrics.set_quality(
            "roundtrip_ratio",
            ratio(
                counters.get("roundtrip_ok", 0),
                counters.get("roundtrip_ok", 0) + counters.get("roundtrip_failed", 0),
            ),
        )

        metrics.meta["expected_quality_metrics"] = [
            "terms_preserved_ratio",
            "formulas_correct_ratio",
            "numunit_merged_ratio",
            "abbreviations_correct_ratio",
            "oov_rate",
            "roundtrip_ratio",
        ]

        metrics.meta["thresholds"] = {
            "terms_preserved_ratio": "1.0",
            "formulas_correct_ratio": "1.0",
            "numunit_merged_ratio": "1.0",
            "abbreviations_correct_ratio": "1.0",
            "oov_rate": "monitor, high values -> extend domain vocab or fix encoding noise",
            "roundtrip_ratio": "1.0",
        }

        metrics.meta["libraries"] = [
            "SentencePiece",
            "HF tokenizers",
            "tiktoken",
            "NLTK",
            "spaCy",
            "SymPy",
        ]