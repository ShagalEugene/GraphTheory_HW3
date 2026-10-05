from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseStage, MetricsCollector, StageContext
from .protect import (
    DISPLAY_MATH_RE,
    INLINE_MATH_RE,
    NUMUNIT_RE,
    TextProtector,
)


@dataclass(frozen=True)
class ChunkingConfig:
    encoding: str = "utf-8"
    token_encoding: str = "cl100k_base"

    chunk_size_tokens: int = 800
    chunk_overlap_percent: float = 0.15

    min_chunk_tokens: int = 512
    max_chunk_tokens: int = 1024

    protect_math: bool = True
    protect_tables: bool = True
    protect_numunit: bool = True

    separators: Tuple[str, ...] = (
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
    )

    output_suffix: str = ".chunks"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


class ChunkingStage(BaseStage):
    name = "chunking"

    _encoder = None
    _encoder_failed = False

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        super().__init__(config or ChunkingConfig())

    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        metrics: MetricsCollector,
    ) -> Path:
        cfg: ChunkingConfig = self.config

        text = input_path.read_text(encoding=cfg.encoding)

        file_metrics: Dict[str, Any] = {
            "chars_before": len(text),
        }

        protector = TextProtector()

        if cfg.protect_math:
            text, count = protector.protect(
                text=text,
                pattern=DISPLAY_MATH_RE,
                kind="display_math",
                block=True,
            )

            if count:
                file_metrics["display_math_protected"] = count

            text, count = protector.protect(
                text=text,
                pattern=INLINE_MATH_RE,
                kind="inline_math",
            )

            if count:
                file_metrics["inline_math_protected"] = count

        if cfg.protect_tables:
            text = self._protect_tables(text, protector, file_metrics)

        if cfg.protect_numunit:
            text, count = protector.protect(
                text=text,
                pattern=NUMUNIT_RE,
                kind="numunit",
            )

            if count:
                file_metrics["numunit_protected"] = count

        overlap_tokens = int(cfg.chunk_size_tokens * cfg.chunk_overlap_percent)

        splitter = RecursiveCharacterTextSplitter(
            separators=list(cfg.separators),
            chunk_size=cfg.chunk_size_tokens,
            chunk_overlap=overlap_tokens,
            length_function=self._make_length_function(protector),
        )

        protected_chunks = splitter.split_text(text)

        chunks = [protector.restore(part).strip() for part in protected_chunks]
        chunks = [chunk for chunk in chunks if chunk]

        chunks, overlap_counts = self._apply_overlap(chunks, overlap_tokens)

        chunks = [self._flatten_block(chunk) for chunk in chunks]
        chunks = [chunk for chunk in chunks if chunk]

        restored_full = protector.restore(text)
        flattened_full = self._flatten_block(restored_full)

        token_counts = [self._count_tokens(chunk) for chunk in chunks]

        sentences = [
            sentence.strip()
            for sentence in SENTENCE_SPLIT_RE.split(flattened_full)
            if len(sentence.strip()) >= 20
        ]

        paragraphs = [
            self._flatten_block(paragraph)
            for paragraph in PARAGRAPH_SPLIT_RE.split(restored_full)
            if paragraph.strip()
        ]
        
        formulas = [
            self._flatten_block(content)
            for kind, content in protector.blocks
            if kind in ("display_math", "inline_math")
        ]

        split_sentences = sum(
            1 for sentence in sentences if not self._contained(sentence, chunks)
        )
        split_paragraphs = sum(
            1 for paragraph in paragraphs if not self._contained(paragraph, chunks)
        )
        split_formulas = sum(
            1 for formula in formulas if not self._contained(formula, chunks)
        )

        file_metrics["chunks_created"] = len(chunks)
        file_metrics["tokens_total"] = sum(token_counts)
        file_metrics["tokens_sum_squares"] = sum(t * t for t in token_counts)
        file_metrics["sentences_total"] = len(sentences)
        file_metrics["split_sentences"] = split_sentences
        file_metrics["paragraphs_total"] = len(paragraphs)
        file_metrics["split_paragraphs"] = split_paragraphs
        file_metrics["formulas_total"] = len(formulas)
        file_metrics["split_formulas"] = split_formulas
        file_metrics["oversized_chunks"] = sum(
            1 for t in token_counts if t > cfg.max_chunk_tokens
        )
        file_metrics["undersized_chunks"] = sum(
            1 for t in token_counts if t < cfg.min_chunk_tokens
        )
        file_metrics["overlap_tokens_total"] = sum(overlap_counts)
        file_metrics["chunks_with_overlap"] = sum(
            1 for value in overlap_counts if value > 0
        )

        record = {
            "source_file": input_path.name,
            "chunk_size_tokens": cfg.chunk_size_tokens,
            "overlap_tokens": overlap_tokens,
            "chunks": [
                {
                    "index": index,
                    "tokens": tokens,
                    "chars": len(chunk),
                    "overlap_prev_tokens": overlap_counts[index],
                    "text": chunk,
                }
                for index, (tokens, chunk) in enumerate(zip(token_counts, chunks))
            ],
        }

        chunks_path = output_dir / f"{input_path.stem}{cfg.output_suffix}.json"
        chunks_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding=cfg.encoding,
        )

        for key, value in file_metrics.items():
            if isinstance(value, int):
                metrics.inc(key, value)

        metrics.add_file_metrics(input_path.name, file_metrics)

        return chunks_path

    def _protect_tables(
        self,
        text: str,
        protector: TextProtector,
        file_metrics: Dict[str, Any],
    ) -> str:
        lines = text.split("\n")
        result: List[str] = []
        block: List[str] = []
        count = 0

        for line in lines:
            if self._is_table_row(line):
                block.append(line)
                continue

            if not line.strip() and block:
                block.append(line)
                continue

            if block:
                if len(block) >= 2:
                    marker = protector.add("markdown_table", "\n".join(block))
                    result.append("")
                    result.append(marker)
                    result.append("")
                    count += 1
                else:
                    result.extend(block)

                block = []

            result.append(line)

        if block:
            if len(block) >= 2:
                marker = protector.add("markdown_table", "\n".join(block))
                result.append("")
                result.append(marker)
                result.append("")
                count += 1
            else:
                result.extend(block)

        if count:
            file_metrics["markdown_tables_protected"] = count

        return "\n".join(result)

    def _is_table_row(self, line: str) -> bool:
        return line.strip().count("|") >= 2

    def _contained(self, needle: str, chunks: List[str]) -> bool:
        for chunk in chunks:
            if needle in chunk:
                return True
        return False

    def _make_length_function(self, protector: TextProtector):
        cache: Dict[str, int] = {}

        def length_function(text: str) -> int:
            cached = cache.get(text)

            if cached is not None:
                return cached

            value = self._count_tokens(protector.restore(text))
            cache[text] = value

            return value

        return length_function

    def _count_tokens(self, text: str) -> int:
        encoder = self._get_encoder()

        if encoder is None:
            return max(1, len(text) // 4)

        return len(encoder.encode(text, disallowed_special=()))

    def _get_encoder(self):
        if ChunkingStage._encoder is None and not ChunkingStage._encoder_failed:
            try:
                ChunkingStage._encoder = tiktoken.get_encoding(
                    self.config.token_encoding
                )
            except Exception:
                ChunkingStage._encoder_failed = True

        return ChunkingStage._encoder

    def _apply_overlap(
        self,
        chunks: List[str],
        overlap_tokens: int,
    ) -> Tuple[List[str], List[int]]:
        if overlap_tokens <= 0 or len(chunks) < 2:
            return chunks, [0] * len(chunks)

        max_overlap_chars = overlap_tokens * 6
        result: List[str] = [chunks[0]]
        overlaps: List[int] = [0]

        for index in range(1, len(chunks)):
            previous = chunks[index - 1]
            current = chunks[index]

            cut = max(0, len(previous) - max_overlap_chars)

            paragraph_starts = [
                match.end()
                for match in PARAGRAPH_SPLIT_RE.finditer(previous, cut)
            ]
            sentence_starts = [
                match.start()
                for match in SENTENCE_SPLIT_RE.finditer(previous, cut)
            ]

            candidates = [
                pos for pos in paragraph_starts + sentence_starts if pos >= cut
            ]
            
            if not candidates:
                result.append(current)
                overlaps.append(0)
                continue

            start = min(candidates)
            span = previous[start:].strip()

            span_tokens = self._count_tokens(span)
            
            if span_tokens > overlap_tokens * 2:
                sentence_cuts = [
                    pos for pos in sentence_starts if pos >= cut
                ]
                if sentence_cuts:
                    for pos in reversed(sentence_cuts):
                        candidate_span = previous[pos:].strip()
                        if self._count_tokens(candidate_span) <= overlap_tokens * 1.2:
                            span = candidate_span
                            span_tokens = self._count_tokens(span)
                            break

            if not span or span in current:
                result.append(current)
                overlaps.append(0)
                continue

            result.append(span + " " + current)
            overlaps.append(span_tokens)

        return result, overlaps
    
    def _flatten_block(self, text: str) -> str:
        parts: List[str] = []
        text_lines: List[str] = []
        table_lines: List[str] = []

        for line in text.split("\n"):
            if self._is_table_row(line):
                if text_lines:
                    parts.append(self._collapse_spaces(" ".join(text_lines)))
                    text_lines = []
                table_lines.append(line.strip())
            else:
                if table_lines:
                    parts.append("\n".join(table_lines))
                    table_lines = []
                if line.strip():
                    text_lines.append(line.strip())

        if text_lines:
            parts.append(self._collapse_spaces(" ".join(text_lines)))

        if table_lines:
            parts.append("\n".join(table_lines))

        return " ".join(part for part in parts if part)

    def _collapse_spaces(self, value: str) -> str:
        return re.sub(r"[ \t]{2,}", " ", value)

    def finalize_metrics(self, metrics: MetricsCollector) -> None:
        counters = metrics.counters

        chunks = counters.get("chunks_created", 0)
        tokens = counters.get("tokens_total", 0)

        avg = tokens / chunks if chunks else 0.0
        sum_squares = counters.get("tokens_sum_squares", 0)
        variance = (sum_squares / chunks - avg * avg) if chunks else 0.0
        std = math.sqrt(max(variance, 0.0))

        sentences = counters.get("sentences_total", 0)
        paragraphs = counters.get("paragraphs_total", 0)
        formulas = counters.get("formulas_total", 0)
        overlap_total = counters.get("overlap_tokens_total", 0)

        metrics.set_quality(
            "avg_overlap_tokens",
            round(overlap_total / chunks, 2) if chunks else 0.0,
        )
        metrics.set_quality(
            "overlap_ratio",
            round(overlap_total / tokens, 4) if tokens else None,
        )
        metrics.set_quality("avg_chunk_tokens", round(avg, 2))
        metrics.set_quality("std_chunk_tokens", round(std, 2))
        metrics.set_quality(
            "broken_sentences_ratio",
            round(counters.get("split_sentences", 0) / sentences, 4)
            if sentences
            else None,
        )
        metrics.set_quality(
            "broken_paragraphs_ratio",
            round(counters.get("split_paragraphs", 0) / paragraphs, 4)
            if paragraphs
            else None,
        )
        metrics.set_quality(
            "broken_formulas_ratio",
            round(counters.get("split_formulas", 0) / formulas, 4)
            if formulas
            else None,
        )
        metrics.set_quality("hit_rate", None)

        metrics.meta["expected_quality_metrics"] = [
            "avg_chunk_tokens",
            "std_chunk_tokens",
            "broken_sentences_ratio",
            "broken_paragraphs_ratio",
            "broken_formulas_ratio",
            "overlap_ratio",
            "hit_rate",
        ]

        metrics.meta["thresholds"] = {
            "chunk_size_tokens": "512-1024",
            "std_chunk_tokens": "<=200",
            "broken_sentences_ratio": "<=0.05",
            "broken_paragraphs_ratio": "<=0.02",
            "broken_formulas_ratio": "0",
            "overlap_ratio": "0.10-0.20",
            "hit_rate": ">0.90",
        }

        metrics.meta["libraries"] = [
            "langchain-text-splitters",
            "tiktoken",
            "sentence-transformers",
            "spaCy",
            "latex2text",
        ]