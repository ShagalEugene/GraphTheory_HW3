from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from dateutil import parser as dateutil_parser
from pint import UnitRegistry
import pymorphy3
from snowballstemmer import stemmer as snowball_stemmer
import sympy
from sympy.parsing.latex import parse_latex
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BaseStage, MetricsCollector
from .protect import (
    DISPLAY_MATH_RE,
    INLINE_MATH_RE,
    NUMUNIT_RE,
    UNIT_VOCAB,
    TextProtector,
)


@dataclass(frozen=True)
class NormalizationConfig:
    encoding: str = "utf-8"

    normalize_units: bool = True
    normalize_dates: bool = True
    normalize_numbers: bool = True
    normalize_formulas: bool = True
    normalize_tables: bool = True

    abbreviations: Dict[str, str] = field(default_factory=dict)
    terms: Tuple[str, ...] = ()

    lemmatization: str = "none"
    preserve_initial_case: bool = True
    protect_mixed_case: bool = True

    extra_pint_definitions: Tuple[str, ...] = ()

    output_suffix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


MONTH_NAMES: Dict[str, int] = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

DATE_DMY_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{4}\b")

DATE_MONTH_NAME_RE = re.compile(
    r"\b(\d{1,2})\s+([A-Za-zА-Яа-яЁё]+)\s+(\d{4})(?:\s*(?:г\.|года))?\b"
)

SCIENTIFIC_RE = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё.])(\d+(?:[.,]\d+)?)\s?[·×x]\s?10\s?(?:\^|\*\*)?\s?(-?\d+)(?![0-9A-Za-zА-Яа-яЁё])"
)

SCIENTIFIC_E_RE = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё.])(\d+(?:[.,]\d+)?)[eE]([+-]?\d{1,3})(?![0-9A-Za-zА-Яа-яЁё])"
)

ELEMENT_SYMBOLS: Tuple[str, ...] = (
    "Nb", "Ti", "V", "Mo", "Cr", "Ni", "Cu", "Mn", "Si", "Al",
    "Ca", "Ta", "Zr", "Hf", "W",
)

GREEK_NAMES: Tuple[str, ...] = (
    "alpha", "beta", "gamma", "delta", "epsilon",
    "sigma", "tau", "omega", "theta", "lambda",
)

CHEMICAL_FORMULA_RE = re.compile(
    r"(?:\[[A-Z][a-z]?\]|[A-Z][a-z]?){2,}"
)

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]*")

TABLE_SPLIT_RE = re.compile(r"(?<!\\)\|")

ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

PLAIN_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d+)?")

STANDALONE_EXCLUDED_UNITS: frozenset = frozenset(
    {"м", "с", "ч", "т", "В", "А", "К", "Н", "л", "%"}
)

STANDALONE_UNIT_ALTERNATION = "|".join(
    sorted(
        (
            re.escape(unit)
            for unit in UNIT_VOCAB
            if unit not in STANDALONE_EXCLUDED_UNITS
        ),
        key=len,
        reverse=True,
    )
)

UNIT_ONLY_RE = re.compile(
    rf"(?<![0-9A-Za-zА-Яа-яЁё])({STANDALONE_UNIT_ALTERNATION})(?![0-9A-Za-zА-Яа-яЁё])"
)

class NormalizationStage(BaseStage):
    name = "normalization"

    _registry = None
    _registry_failed = False
    _morph = None
    _stemmer = None

    def __init__(self, config: Optional[NormalizationConfig] = None) -> None:
        super().__init__(config or NormalizationConfig())

    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        metrics: MetricsCollector,
    ) -> Path:
        cfg: NormalizationConfig = self.config

        text = input_path.read_text(encoding=cfg.encoding)

        file_metrics: Dict[str, Any] = {
            "chars_before": len(text),
        }

        protector = TextProtector()
        tables: List[Dict[str, Any]] = []

        transform: Optional[Callable[[str], str]] = None

        if cfg.normalize_formulas:
            def transform_formula(content: str) -> str:
                if CHEMICAL_FORMULA_RE.search(content):
                    file_metrics["formulas_skipped_chemical"] = (
                        file_metrics.get("formulas_skipped_chemical", 0) + 1
                    )
                    return content

                canonical = self._canonical_formula(content)

                if canonical != content:
                    file_metrics["formulas_canonicalized"] = (
                        file_metrics.get("formulas_canonicalized", 0) + 1
                    )

                return canonical

            transform = transform_formula

        text, count = protector.protect(
            text=text,
            pattern=DISPLAY_MATH_RE,
            kind="display_math",
            block=True,
            transform=transform,
        )

        if count:
            file_metrics["display_math_protected"] = count

        text, count = protector.protect(
            text=text,
            pattern=INLINE_MATH_RE,
            kind="inline_math",
            transform=transform,
        )

        if count:
            file_metrics["inline_math_protected"] = count

        if cfg.normalize_tables:
            text = self._protect_and_normalize_tables(
                text=text,
                protector=protector,
                file_metrics=file_metrics,
                tables=tables,
                source_name=input_path.name,
            )

        text = self._normalize_dates(text, file_metrics)
        text = self._normalize_numbers(text, file_metrics)
        text = self._normalize_units(text, file_metrics)
        text = self._normalize_abbreviations(text, file_metrics)
        text = self._lemmatize(text, file_metrics)

        text = protector.restore(text)
        for record in tables:
            self._restore_record(record, protector)

        if tables:
            tables_dir = output_dir / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)

            tables_path = tables_dir / f"{input_path.stem}.tables.json"
            tables_path.write_text(
                json.dumps(tables, ensure_ascii=False, indent=2),
                encoding=cfg.encoding,
            )

            file_metrics["tables_exported"] = len(tables)

        if text and not text.endswith("\n"):
            text += "\n"

        file_metrics["chars_after"] = len(text)

        for key, value in file_metrics.items():
            if isinstance(value, int):
                metrics.inc(key, value)

        metrics.add_file_metrics(input_path.name, file_metrics)

        out_name = input_path.stem + cfg.output_suffix + input_path.suffix
        output_path = output_dir / out_name

        output_path.write_text(text, encoding=cfg.encoding)

        return output_path

    def _protect_and_normalize_tables(
        self,
        text: str,
        protector: TextProtector,
        file_metrics: Dict[str, Any],
        tables: List[Dict[str, Any]],
        source_name: str,
    ) -> str:
        lines = text.split("\n")
        result: List[str] = []
        block: List[str] = []
        table_count = 0
        typed_columns = 0

        for line in lines:
            if self._is_table_row(line):
                block.append(line)
                continue
            
            if not line.strip() and block:
                block.append(line)
                continue

            if block:
                if len(block) >= 2:
                    normalized, typed, record = self._normalize_markdown_table(
                        block="\n".join(block),
                        file_metrics=file_metrics,
                        source_name=source_name,
                        table_index=table_count,
                    )
                    marker = protector.add("markdown_table", normalized)
                    result.append("")
                    result.append(marker)
                    result.append("")
                    table_count += 1
                    typed_columns += typed

                    if record is not None:
                        tables.append(record)
                else:
                    result.extend(block)

                block = []

            result.append(line)

        if block:
            if len(block) >= 2:
                normalized, typed, record = self._normalize_markdown_table(
                    block="\n".join(block),
                    file_metrics=file_metrics,
                    source_name=source_name,
                    table_index=table_count,
                )
                marker = protector.add("markdown_table", normalized)
                result.append("")
                result.append(marker)
                result.append("")
                table_count += 1
                typed_columns += typed

                if record is not None:
                    tables.append(record)
            else:
                result.extend(block)

        if table_count:
            file_metrics["markdown_tables_normalized"] = (
                file_metrics.get("markdown_tables_normalized", 0) + table_count
            )

        if typed_columns:
            file_metrics["table_typed_columns"] = (
                file_metrics.get("table_typed_columns", 0) + typed_columns
            )

        return "\n".join(result)

    def _normalize_markdown_table(
        self,
        block: str,
        file_metrics: Dict[str, Any],
        source_name: str,
        table_index: int,
    ) -> Tuple[str, int, Optional[Dict[str, Any]]]:
        parsed: List[Tuple[bool, List[str]]] = []

        for line in block.split("\n"):
            if not line.strip():
                continue

            if self._is_table_separator(line):
                parsed.append((True, self._split_table_row(line)))
            else:
                parsed.append((False, self._split_table_row(line)))

        if not parsed:
            return block, 0, None

        separator_positions = [
            index
            for index, (is_separator, _) in enumerate(parsed)
            if is_separator
        ]

        if separator_positions:
            separator = separator_positions[0]
            header_rows = [
                cells for is_separator, cells in parsed[:separator] if not is_separator
            ]
            header = header_rows[0] if header_rows else []
            body = [
                cells for is_separator, cells in parsed[separator + 1:] if not is_separator
            ]
        else:
            header = parsed[0][1]
            body = [cells for is_separator, cells in parsed[1:] if not is_separator]

        if not header:
            width = max([len(cells) for _, cells in parsed] + [1])
            header = [f"col_{i}" for i in range(width)]

        width = max([len(header)] + [len(cells) for cells in body])
        header = header + [""] * (width - len(header))
        body = [cells + [""] * (width - len(cells)) for cells in body]

        header = [
            self._normalize_table_cell(cell, file_metrics)[0]
            for cell in header
        ]

        display_body: List[List[str]] = []
        structured_body: List[List[Any]] = []

        for row in body:
            display_row: List[str] = []
            structured_row: List[Any] = []

            for cell in row:
                display, structured = self._normalize_table_cell(cell, file_metrics)
                display_row.append(display)
                structured_row.append(structured)

            display_body.append(display_row)
            structured_body.append(structured_row)

        column_specs = self._infer_column_specs(header, structured_body)

        lines_out = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]

        lines_out.extend(
            "| " + " | ".join(row) + " |"
            for row in display_body
        )

        typed_columns = 0
        record: Optional[Dict[str, Any]] = None

        if pd is not None and structured_body:
            frame = pd.DataFrame(
                {spec["name"]: list(spec["values"]) for spec in column_specs}
            )

            for spec in column_specs:
                if spec["dtype"] == "float64":
                    frame[spec["name"]] = frame[spec["name"]].astype("float64")
                    typed_columns += 1
                elif spec["dtype"] == "datetime64[ns]":
                    frame[spec["name"]] = pd.to_datetime(
                        frame[spec["name"]],
                        errors="coerce",
                    )
                    typed_columns += 1

            export_frame = frame.copy()

            for spec in column_specs:
                if spec["dtype"] == "datetime64[ns]":
                    export_frame[spec["name"]] = export_frame[spec["name"]].dt.strftime("%Y-%m-%d")

            export_frame = export_frame.astype(object).where(export_frame.notna(), None)

            record = {
                "source_file": source_name,
                "table_index": table_index,
                "shape": {
                    "rows": len(structured_body),
                    "columns": width,
                },
                "columns": [
                    {
                        "name": spec["name"],
                        "dtype": str(frame[spec["name"]].dtype),
                        "unit": spec["unit"],
                    }
                    for spec in column_specs
                ],
                "rows": export_frame.to_dict(orient="records"),
            }

        return "\n".join(lines_out), typed_columns, record

    def _split_table_row(self, line: str) -> List[str]:
        stripped = line.strip()

        if stripped.startswith("|"):
            stripped = stripped[1:]

        if stripped.endswith("|"):
            stripped = stripped[:-1]

        parts = TABLE_SPLIT_RE.split(stripped)

        return [
            part.strip().replace("\\|", "|")
            for part in parts
        ]

    def _is_table_separator(self, line: str) -> bool:
        stripped = line.strip()

        if "|" not in stripped:
            return False

        cleaned = (
            stripped
            .replace("|", " ")
            .replace(":", "")
            .replace("-", "")
            .replace(" ", "")
        )

        return cleaned == "" and "-" in stripped

    def _normalize_table_cell(
        self,
        value: str,
        file_metrics: Dict[str, Any],
    ) -> Tuple[str, Any]:
        stripped = value.strip()

        if not stripped:
            return "", None

        display = self._normalize_dates(stripped, file_metrics)
        display = self._normalize_numbers(display, file_metrics)

        quantity = self._convert_quantity(display, file_metrics)

        if quantity is not None:
            magnitude, symbol = quantity
            display = f"{self._format_number(magnitude)} {symbol}"

            return display, {
                "kind": "quantity",
                "value": magnitude,
                "unit": symbol,
            }

        display = self._normalize_units(display, file_metrics).strip()

        number = self._parse_plain_number(display)

        if number is not None:
            return display, {"kind": "number", "value": number}

        if ISO_DATE_RE.fullmatch(display):
            return display, {"kind": "date", "value": display}

        return display, {"kind": "text", "value": display}

    def _convert_quantity(self, text: str, file_metrics: Dict[str, Any]):
        cfg: NormalizationConfig = self.config

        if not cfg.normalize_units:
            return None

        registry = self._get_registry()

        if registry is None:
            return None

        match = NUMUNIT_RE.fullmatch(text)

        if match is None:
            return None

        target = UNIT_VOCAB.get(match.group(2))

        if target is None:
            return None

        raw_value = match.group(1).replace(",", ".")

        try:
            quantity = registry(f"{raw_value} {target}")
            si = quantity.to_base_units()
        except Exception:
            return None

        file_metrics["units_normalized"] = (
            file_metrics.get("units_normalized", 0) + 1
        )
        symbol = re.sub(r"\s*([*/])\s*", r"\1", f"{si.units:~}")
        return float(si.magnitude), symbol

    def _parse_plain_number(self, display: str) -> Optional[float]:
        if PLAIN_NUMBER_RE.fullmatch(display) is None:
            return None

        try:
            return float(display.replace(",", "."))
        except ValueError:
            return None

    def _infer_column_specs(
        self,
        header: List[str],
        structured_body: List[List[Any]],
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        used_names: set = set()

        for index, raw_name in enumerate(header):
            name = raw_name.strip() or f"col_{index}"

            if name in used_names:
                name = f"{name}_{index}"

            used_names.add(name)

            entries = [row[index] for row in structured_body]
            non_empty = [entry for entry in entries if entry is not None]

            dtype = "object"
            unit: Optional[str] = None
            values: List[Any] = [
                None if entry is None else str(entry["value"])
                for entry in entries
            ]

            if non_empty:
                kinds = {entry["kind"] for entry in non_empty}

                if kinds <= {"quantity", "number"}:
                    units = {
                        entry["unit"]
                        for entry in non_empty
                        if entry["kind"] == "quantity"
                    }

                    if len(units) <= 1:
                        dtype = "float64"
                        unit = next(iter(units), None)
                        values = [
                            None if entry is None else entry["value"]
                            for entry in entries
                        ]
                elif kinds == {"date"}:
                    dtype = "datetime64[ns]"
                    values = [
                        None if entry is None else entry["value"]
                        for entry in entries
                    ]

            specs.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "unit": unit,
                    "values": values,
                }
            )

        return specs

    def _is_table_row(self, line: str) -> bool:
        stripped = line.strip()

        if stripped.count("|") < 2:
            return False

        return True

    def _canonical_formula(self, latex: str) -> str:
        if parse_latex is None or sympy is None:
            return latex

        if latex.startswith("$$"):
            prefix, suffix, inner = "$$", "$$", latex[2:-2]
        elif latex.startswith("\\["):
            prefix, suffix, inner = "\\[", "\\]", latex[2:-2]
        elif latex.startswith("\\("):
            prefix, suffix, inner = "\\(", "\\)", latex[2:-2]
        else:
            prefix, suffix, inner = "$", "$", latex[1:-1]

        if CHEMICAL_FORMULA_RE.search(inner):
            return latex

        try:
            expr = parse_latex(inner)
            canonical = sympy.latex(expr)
        except Exception:
            return latex

        if not self._notation_preserved(inner, canonical):
            return latex

        return f"{prefix}{canonical}{suffix}"

    def _notation_preserved(self, original: str, canonical: str) -> bool:
        for symbol in ELEMENT_SYMBOLS:
            if symbol in original and symbol not in canonical:
                return False

        for name in GREEK_NAMES:
            token = f"\\{name}"
            if token in original and token not in canonical:
                return False

        return False if not canonical.strip() else True

    def _normalize_dates(self, text: str, file_metrics: Dict[str, Any]) -> str:
        cfg: NormalizationConfig = self.config

        if not cfg.normalize_dates:
            return text

        count = 0

        def repl_numeric(match: re.Match) -> str:
            nonlocal count

            if dateutil_parser is not None:
                try:
                    parsed = dateutil_parser.parse(
                        match.group(0),
                        dayfirst=True,
                    )
                    count += 1
                    return parsed.date().isoformat()
                except Exception:
                    return match.group(0)

            day, month, year = (
                int(part) for part in re.split(r"[./]", match.group(0))
            )

            try:
                count += 1
                return date(year, month, day).isoformat()
            except ValueError:
                return match.group(0)

        text = DATE_DMY_RE.sub(repl_numeric, text)

        def repl_named(match: re.Match) -> str:
            nonlocal count

            month = MONTH_NAMES.get(match.group(2).lower())

            if month is None:
                return match.group(0)

            try:
                parsed = date(
                    int(match.group(3)),
                    month,
                    int(match.group(1)),
                )
            except ValueError:
                return match.group(0)

            count += 1

            return parsed.isoformat()

        text = DATE_MONTH_NAME_RE.sub(repl_named, text)

        if count:
            file_metrics["dates_normalized"] = (
                file_metrics.get("dates_normalized", 0) + count
            )

        return text

    def _normalize_numbers(self, text: str, file_metrics: Dict[str, Any]) -> str:
        cfg: NormalizationConfig = self.config

        if not cfg.normalize_numbers:
            return text

        count_scientific = 0

        def repl_scientific(match: re.Match) -> str:
            nonlocal count_scientific

            mantissa = match.group(1).replace(",", ".")

            try:
                value = float(mantissa) * (10 ** int(match.group(2)))
            except Exception:
                return match.group(0)

            count_scientific += 1

            return self._format_number(value)

        text = SCIENTIFIC_RE.sub(repl_scientific, text)
        text = SCIENTIFIC_E_RE.sub(repl_scientific, text)

        if count_scientific:
            file_metrics["scientific_numbers_normalized"] = (
                file_metrics.get("scientific_numbers_normalized", 0) + count_scientific
            )

        count_dot_to_comma = 0
        
        def repl_dot(match: re.Match) -> str:
            nonlocal count_dot_to_comma
            count_dot_to_comma += 1
            return f"{match.group(1)},{match.group(2)}"

        text = re.sub(
            r"(?<![0-9A-Za-zА-Яа-яЁё.,-])(\d+)\.(\d+)(?![0-9A-Za-zА-Яа-яЁё.,-])",
            repl_dot,
            text
        )

        if count_dot_to_comma:
            file_metrics["decimal_dots_to_commas"] = (
                file_metrics.get("decimal_dots_to_commas", 0) + count_dot_to_comma
            )

        return text

    def _format_number(self, value: float) -> str:
        if value == 0:
            return "0"

        if abs(value) >= 1e15 or abs(value) < 1e-9:
            return f"{value:e}".replace(".", ",")

        formatted = f"{value:.10f}".rstrip("0").rstrip(".")
        return (formatted or "0").replace(".", ",")
    
    def _normalize_units(self, text: str, file_metrics: Dict[str, Any]) -> str:
        cfg: NormalizationConfig = self.config

        if not cfg.normalize_units:
            return text

        registry = self._get_registry()

        if registry is None:
            return text

        count = 0

        def repl(match: re.Match) -> str:
            nonlocal count

            raw_value = match.group(1).replace(",", ".")
            unit_key = match.group(2)

            target = UNIT_VOCAB.get(unit_key)

            if target is None:
                return match.group(0)

            try:
                quantity = registry(f"{raw_value} {target}")
                si = quantity.to_base_units()
            except Exception:
                return match.group(0)

            count += 1

            magnitude = self._format_number(float(si.magnitude))
            symbol = re.sub(r"\s*([*/])\s*", r"\1", f"{si.units:~}")

            return f"{magnitude} {symbol}"

        text = NUMUNIT_RE.sub(repl, text)
        text = self._normalize_standalone_units(text, file_metrics)

        if count:
            file_metrics["units_normalized"] = (
                file_metrics.get("units_normalized", 0) + count
            )

        return text

    def _normalize_abbreviations(self, text: str, file_metrics: Dict[str, Any]) -> str:
        cfg: NormalizationConfig = self.config

        if not cfg.abbreviations:
            return text

        count = 0

        for source, target in cfg.abbreviations.items():
            pattern = re.compile(
                rf"(?<![\w]){re.escape(source)}(?![\w])"
            )
            text, replaced = pattern.subn(target, text)
            count += replaced

        if count:
            file_metrics["abbreviations_normalized"] = (
                file_metrics.get("abbreviations_normalized", 0) + count
            )

        return text

    def _lemmatize(self, text: str, file_metrics: Dict[str, Any]) -> str:
        cfg: NormalizationConfig = self.config

        mode = cfg.lemmatization

        if mode == "none":
            return text

        terms = set(cfg.terms)
        count = 0
        case_preserved = 0

        def is_protected(token: str) -> bool:
            if token in terms:
                return True

            if token.isupper():
                return True

            if len(token) < 3:
                return True

            if cfg.protect_mixed_case and any(ch.isupper() for ch in token[1:]):
                return True

            return False

        def apply_case(token: str, normal: str) -> str:
            nonlocal case_preserved

            if not cfg.preserve_initial_case:
                return normal

            if not token[0].isupper():
                return normal

            case_preserved += 1

            return normal.capitalize()

        if mode == "lemmatize":
            analyzer = self._get_morph()

            if analyzer is None:
                file_metrics["lemmatization_analyzer_unavailable"] = 1
                return text

            def repl(match: re.Match) -> str:
                nonlocal count

                token = match.group(0)

                if is_protected(token):
                    return token

                try:
                    normal = analyzer.parse(token)[0].normal_form
                except Exception:
                    return token

                normal = apply_case(token, normal)

                if normal != token:
                    count += 1

                return normal

            text = TOKEN_RE.sub(repl, text)

        elif mode == "stem":
            stemmer = self._get_stemmer()

            if stemmer is None:
                file_metrics["stemming_analyzer_unavailable"] = 1
                return text

            def repl_stem(match: re.Match) -> str:
                nonlocal count

                token = match.group(0)

                if is_protected(token):
                    return token

                stemmed = apply_case(token, stemmer.stemWord(token))

                if stemmed != token:
                    count += 1

                return stemmed

            text = TOKEN_RE.sub(repl_stem, text)

        else:
            return text

        if count:
            file_metrics["tokens_lemmatized"] = (
                file_metrics.get("tokens_lemmatized", 0) + count
            )

        if case_preserved:
            file_metrics["tokens_case_preserved"] = (
                file_metrics.get("tokens_case_preserved", 0) + case_preserved
            )

        return text

    def _get_registry(self):
        cfg: NormalizationConfig = self.config

        if UnitRegistry is None:
            return None

        if NormalizationStage._registry is None and not NormalizationStage._registry_failed:
            try:
                registry = UnitRegistry()

                for definition in cfg.extra_pint_definitions:
                    registry.define(definition)

                NormalizationStage._registry = registry
            except Exception:
                NormalizationStage._registry_failed = True

        return NormalizationStage._registry

    def _get_morph(self):
        if NormalizationStage._morph is None:
            try:
                NormalizationStage._morph = pymorphy3.MorphAnalyzer()
            except Exception:
                return None

        return NormalizationStage._morph

    def _get_stemmer(self):
        if NormalizationStage._stemmer is None:
            try:
                NormalizationStage._stemmer = snowball_stemmer("russian")
            except Exception:
                return None

        return NormalizationStage._stemmer

    def _si_symbol_for(self, target: str) -> Optional[str]:
        registry = self._get_registry()

        if registry is None:
            return None

        try:
            res = f"{registry(f'1 {target}').to_base_units().units:~}"
            return re.sub(r"\s*([*/])\s*", r"\1", f"{res}")
        except Exception:
            return None

    def _normalize_standalone_units(
        self,
        text: str,
        file_metrics: Dict[str, Any],
    ) -> str:
        cfg: NormalizationConfig = self.config

        if not cfg.normalize_units:
            return text

        count = 0

        def repl(match: re.Match) -> str:
            nonlocal count

            target = UNIT_VOCAB.get(match.group(1))

            if target is None:
                return match.group(0)

            symbol = self._si_symbol_for(target)

            if symbol is None:
                return match.group(0)

            count += 1

            return symbol

        text = UNIT_ONLY_RE.sub(repl, text)

        if count:
            file_metrics["standalone_units_normalized"] = (
                file_metrics.get("standalone_units_normalized", 0) + count
            )

        return text

    def _restore_record(
        self,
        record: Dict[str, Any],
        protector: TextProtector,
    ) -> Dict[str, Any]:
        record["columns"] = [
            {
                **spec,
                "name": protector.restore(spec["name"]),
            }
            for spec in record["columns"]
        ]

        record["rows"] = [
            {
                protector.restore(key): (
                    protector.restore(value)
                    if isinstance(value, str)
                    else value
                )
                for key, value in row.items()
            }
            for row in record["rows"]
        ]

        return record

    def finalize_metrics(self, metrics: MetricsCollector) -> None:
        quality_names = [
            "lemmatization_accuracy",
            "unit_correctness",
            "date_iso_accuracy",
            "abbreviation_accuracy",
            "numbers_saved",
        ]

        for name in quality_names:
            metrics.set_quality(name, None)

        metrics.meta["quality_note"] = (
            "Качественные метрики нормализации пока заглушки. "
            "Их можно реализовать через BaseQualityMetric."
        )

        metrics.meta["expected_quality_metrics"] = quality_names

        metrics.meta["libraries"] = [
            "pymorphy2",
            "Mystem",
            "spaCy",
            "Pint",
            "quantities",
            "dateutil",
            "pandas",
            "sympy",
            "latex2sympy",
            "rdkit",
        ]

        metrics.meta["note"] = (
            "Химические формулы (rdkit) и AST-нормализация (latex2sympy) "
            "подключаются как отдельные трансформы, когда в корпусе появится химия."
        )