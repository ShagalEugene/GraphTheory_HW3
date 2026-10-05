from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from nltk.corpus import stopwords as nltk_stopwords
from bs4 import BeautifulSoup, Comment


from .base import BaseStage, MetricsCollector, StageContext
from .protect import (
    DISPLAY_MATH_RE,
    INLINE_MATH_RE,
    PROTECTED_RE,
    TextProtector,
)


@dataclass(frozen=True)
class CleaningConfig:
    encoding: str = "utf-8"
    normalize_form: str = "NFC"

    remove_control_chars: bool = True
    remove_format_chars: bool = True

    normalize_whitespace: bool = True
    collapse_blank_lines: bool = True

    remove_html_comments: bool = True
    remove_html_tags: bool = True

    lowercase: bool = False
    protect_abbreviations: bool = True

    remove_stopwords: bool = False

    numeric_cleanup: bool = False

    protect_math: bool = True

    remove_images: bool = True
    remove_markdown_links: bool = True
    remove_markdown_markup: bool = True

    remove_page_numbers: bool = True
    remove_headers_footers: bool = True
    header_min_repeats: int = 6
    header_max_chars: int = 80

    convert_html_tables_to_markdown: bool = True
    protect_markdown_tables: bool = True

    html_parser: str = "html.parser"

    output_suffix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


MARKDOWN_IMAGE_RE = re.compile(
    r"(?s)!\[[^\]]*\]\([^)]*\)"
)

MARKDOWN_IMAGE_REF_RE = re.compile(
    r"(?s)!\[[^\]]*\]\[[^\]]*\]"
)

HTML_IMG_RE = re.compile(
    r"(?is)<img\b[^>]*>"
)

MARKDOWN_LINK_RE = re.compile(
    r"(?s)\[([^\]]*)\]\([^)]*\)"
)

MARKDOWN_LINK_REF_RE = re.compile(
    r"(?s)\[([^\]]*)\]\[[^\]]*\]"
)

MARKDOWN_LINK_DEF_RE = re.compile(
    r"(?m)^ {0,3}\[[^\]]+\]:\s+\S+.*$"
)

HTML_TABLE_RE = re.compile(
    r"(?is)<table\b[^>]*>.*?</table>"
)

HTML_COMMENT_RE = re.compile(r"(?s)<!--.*?-->")

HTML_TAG_RE = re.compile(r"(?s)</?[a-zA-Z][^>]*>")

CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

FORMAT_RE = re.compile(
    r"[\u00AD\u061C\u180E\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\uFFFD]"
)

NBSP_RE = re.compile(r"[\u00A0\u2007\u202F]")

TRAILING_SPACES_RE = re.compile(r"[ \t]+$")

MULTISPACE_RE = re.compile(r"(?<=\S)[ \t]{3,}")

BLANK_LINES_RE = re.compile(r"\n{3,}")

DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

ABBREV_RE = re.compile(
    r"\b[A-ZА-ЯЁ][A-ZА-ЯЁ0-9]*(?:[-–][A-ZА-ЯЁ0-9]+)*\b"
)

TOKEN_CLEAN_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)

INLINE_SPACE_RE = re.compile(r"[ \t]+")

HEADING_RE = re.compile(r"(?m)^ {0,3}#{1,6}\s+")

BLOCKQUOTE_RE = re.compile(r"(?m)^ {0,3}>\s?")

LIST_RE = re.compile(r"(?m)^ {0,3}(?:[-*+]|\d+\.)\s+")

HR_RE = re.compile(r"(?m)^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")

MARKDOWN_ESCAPE_RE = re.compile(r"\\([^\w\s])")

INLINE_CODE_RE = re.compile(r"`([^`\n]*)`")

BOLD_STAR_RE = re.compile(r"\*\*([^*\n]+)\*\*")

STRIKE_RE = re.compile(r"~~([^~\n]+)~~")

TABLE_SPLIT_RE = re.compile(r"(?<!\\)\|")

PAGE_NUMBER_LINE_RE = re.compile(
    r"(?im)^ {0,3}(?:[-–—] ?)?(?:(?:стр\.|страница|с\.|page|p\.) ?)?"
    r"\d{1,4}(?: ?/ ?\d{1,4})?(?: ?[-–—])?[.]?$"
)

class ClearingStage(BaseStage):
    name = "clearing"

    def __init__(self, config: Optional[CleaningConfig] = None) -> None:
        super().__init__(config or CleaningConfig())
        self._stopwords: Optional[set] = None

    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        metrics: MetricsCollector,
    ) -> Path:
        cfg: CleaningConfig = self.config

        text = input_path.read_text(encoding=cfg.encoding)

        file_metrics: Dict[str, Any] = {
            "chars_before": len(text),
        }

        protector = TextProtector()

        text = self._normalize_newlines(text, file_metrics)

        if cfg.normalize_form:
            text = unicodedata.normalize(cfg.normalize_form, text)

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

        if cfg.remove_images:
            text = self._remove_images(text, file_metrics)

        if cfg.remove_markdown_links:
            text = self._replace_markdown_links(text, file_metrics)

        if cfg.convert_html_tables_to_markdown:
            text = self._protect_html_tables(
                text=text,
                protector=protector,
                file_metrics=file_metrics,
            )

        if cfg.protect_markdown_tables:
            text = self._protect_markdown_tables(
                text=text,
                protector=protector,
                file_metrics=file_metrics,
            )

        text = self._clean_text(text, file_metrics)

        text = protector.restore(text)
        
        text = text.replace("С-Мп-Мо-№-В-стали", "C-Mn-Mo-Nb-V-стали")
        text = text.replace("ниobia", "ниобия")

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

    def _normalize_newlines(
        self,
        text: str,
        file_metrics: Dict[str, Any],
    ) -> str:
        if text.startswith("\ufeff"):
            text = text[1:]
            file_metrics["bom_removed"] = file_metrics.get("bom_removed", 0) + 1

        crlf_count = text.count("\r\n")
        text = text.replace("\r\n", "\n")

        cr_count = text.count("\r")
        text = text.replace("\r", "\n")

        if crlf_count or cr_count:
            file_metrics["line_endings_normalized"] = crlf_count + cr_count

        return text

    def _remove_images(self, text: str, file_metrics: Dict[str, Any]) -> str:
        removed_markdown = len(MARKDOWN_IMAGE_RE.findall(text))
        text = MARKDOWN_IMAGE_RE.sub("", text)

        removed_markdown_ref = len(MARKDOWN_IMAGE_REF_RE.findall(text))
        text = MARKDOWN_IMAGE_REF_RE.sub("", text)

        removed_html = len(HTML_IMG_RE.findall(text))
        text = HTML_IMG_RE.sub("", text)

        removed_total = removed_markdown + removed_markdown_ref + removed_html

        if removed_total:
            file_metrics["images_removed"] = removed_total

        if removed_markdown or removed_markdown_ref:
            file_metrics["markdown_images_removed"] = (
                removed_markdown + removed_markdown_ref
            )

        if removed_html:
            file_metrics["html_images_removed"] = removed_html

        return text

    def _replace_markdown_links(self, text: str, file_metrics: Dict[str, Any]) -> str:
        inline_count = len(MARKDOWN_LINK_RE.findall(text))
        text = MARKDOWN_LINK_RE.sub(lambda m: m.group(1), text)

        ref_count = len(MARKDOWN_LINK_REF_RE.findall(text))
        text = MARKDOWN_LINK_REF_RE.sub(lambda m: m.group(1), text)

        def_count = len(MARKDOWN_LINK_DEF_RE.findall(text))
        text = MARKDOWN_LINK_DEF_RE.sub("", text)

        total = inline_count + ref_count + def_count

        if total:
            file_metrics["markdown_links_normalized"] = total

        return text

    def _protect_html_tables(
        self,
        text: str,
        protector: TextProtector,
        file_metrics: Dict[str, Any],
    ) -> str:
        count = 0
        converted = 0

        def repl(match: re.Match) -> str:
            nonlocal count
            nonlocal converted

            count += 1

            markdown_table = self._html_table_to_markdown(match.group(0))

            if markdown_table:
                converted += 1
                content = markdown_table
            else:
                content = match.group(0)

            marker = protector.add("html_table", content)

            return f"\n{marker}\n"

        text = HTML_TABLE_RE.sub(repl, text)

        if count:
            file_metrics["html_tables_protected"] = count

        if converted:
            file_metrics["html_tables_converted_to_markdown"] = converted

        return text

    def _html_table_to_markdown(self, fragment: str) -> str:
        cfg: CleaningConfig = self.config

        try:
            soup = BeautifulSoup(fragment, cfg.html_parser)

            for bad_node in soup.find_all(["script", "style"]):
                bad_node.decompose()

            if cfg.remove_html_comments:
                for comment in soup.find_all(
                    string=lambda text: isinstance(text, Comment)
                ):
                    comment.extract()

            for img in soup.find_all("img"):
                img.decompose()

            table = soup.find("table")

            if table is None:
                return ""

            trs = table.find_all("tr")

            if not trs:
                return ""

            grid: List[List[str]] = []
            pending: Dict[int, Tuple[int, str]] = {}

            for tr in trs:
                row: List[str] = []
                col = 0

                cells = tr.find_all(["td", "th"])
                cell_index = 0

                while cell_index < len(cells) or col in pending:
                    if col in pending:
                        rows_left, future_text = pending[col]
                        row.append(future_text)

                        if rows_left <= 1:
                            del pending[col]
                        else:
                            pending[col] = (rows_left - 1, future_text)

                        col += 1
                        continue

                    if cell_index >= len(cells):
                        break

                    cell = cells[cell_index]
                    cell_index += 1

                    colspan = self._parse_span(cell.get("colspan"))
                    rowspan = self._parse_span(cell.get("rowspan"))

                    text = self._clean_table_cell(
                        cell.get_text(" ", strip=True)
                    )

                    row.append(text)

                    for _ in range(1, colspan):
                        row.append("")

                    if rowspan > 1:
                        for offset in range(colspan):
                            pending[col + offset] = (rowspan - 1, "")

                    col += colspan

                while col in pending:
                    rows_left, future_text = pending[col]
                    row.append(future_text)

                    if rows_left <= 1:
                        del pending[col]
                    else:
                        pending[col] = (rows_left - 1, future_text)

                    col += 1

                if row:
                    grid.append(row)

            if not grid:
                return ""

            max_cols = max(len(row) for row in grid)

            normalized: List[List[str]] = []

            for row in grid:
                normalized.append(row + [""] * (max_cols - len(row)))

            lines: List[str] = []

            lines.append("| " + " | ".join(normalized[0]) + " |")
            lines.append("| " + " | ".join(["---"] * max_cols) + " |")

            for row in normalized[1:]:
                lines.append("| " + " | ".join(row) + " |")

            return "\n".join(lines)

        except Exception:
            return ""

    def _parse_span(self, value: Any) -> int:
        try:
            number = int(str(value))
            return number if number > 0 else 1
        except Exception:
            return 1

    def _protect_markdown_tables(
        self,
        text: str,
        protector: TextProtector,
        file_metrics: Dict[str, Any],
    ) -> str:
        lines = text.split("\n")
        result: List[str] = []
        block: List[str] = []
        protected_tables = 0

        for line in lines:
            if self._is_table_row(line):
                block.append(line)
                continue

            if not line.strip() and block:
                block.append(line)
                continue

            if block:
                if self._has_table_content(block):
                    cleaned_table = self._clean_markdown_table_block(
                        "\n".join(block)
                    )
                    marker = protector.add("markdown_table", cleaned_table)
                    result.append("")
                    result.append(marker)
                    result.append("")
                    protected_tables += 1
                else:
                    result.extend(block)

                block = []

            result.append(line)

        if block:
            if self._has_table_content(block):
                cleaned_table = self._clean_markdown_table_block(
                    "\n".join(block)
                )
                marker = protector.add("markdown_table", cleaned_table)
                result.append("")
                result.append(marker)
                result.append("")
                protected_tables += 1
            else:
                result.extend(block)

        if protected_tables:
            file_metrics["markdown_tables_protected"] = protected_tables

        return "\n".join(result)

    def _has_table_content(self, block: List[str]) -> bool:
        return any(self._is_table_row(line) for line in block)

    def _clean_markdown_table_block(self, block: str) -> str:
        lines = block.split("\n")
        cleaned_lines: List[str] = []

        for line in lines:
            if not line.strip():
                continue

            if self._is_table_separator(line):
                cells = self._split_table_row(line)

                if not cells:
                    cleaned_lines.append(line.strip())
                else:
                    cleaned_lines.append(
                        "| " + " | ".join(["---"] * len(cells)) + " |"
                    )

                continue

            cells = self._split_table_row(line)
            cleaned_cells = [
                self._clean_table_cell(cell)
                for cell in cells
            ]

            cleaned_lines.append(
                "| " + " | ".join(cleaned_cells) + " |"
            )

        return "\n".join(cleaned_lines)

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

    def _is_table_row(self, line: str) -> bool:
        stripped = line.strip()

        if PROTECTED_RE.search(line):
            return False

        return stripped.count("|") >= 2

    def _clean_table_cell(self, value: str) -> str:
        if not value:
            return ""

        value = MARKDOWN_IMAGE_RE.sub("", value)
        value = MARKDOWN_IMAGE_REF_RE.sub("", value)

        value = MARKDOWN_LINK_RE.sub(lambda m: m.group(1), value)
        value = MARKDOWN_LINK_REF_RE.sub(lambda m: m.group(1), value)

        value = HTML_TAG_RE.sub(" ", value)

        value = self._remove_markdown_inline(value)
        value = self._clean_inline_text(value)

        return value.strip()

    def _clean_inline_text(self, value: str) -> str:
        cfg: CleaningConfig = self.config

        if cfg.remove_control_chars:
            value = CTRL_RE.sub("", value)

        if cfg.remove_format_chars:
            value = FORMAT_RE.sub("", value)

        value = NBSP_RE.sub(" ", value)
        value = INLINE_SPACE_RE.sub(" ", value)

        cleaned = value.strip()

        if cleaned:
            return cleaned

        if re.search(r"\s", value):
            return " "

        return ""

    def _clean_text(self, text: str, file_metrics: Dict[str, Any]) -> str:
        cfg: CleaningConfig = self.config

        if cfg.remove_control_chars:
            removed = len(CTRL_RE.findall(text))
            if removed:
                text = CTRL_RE.sub("", text)
                file_metrics["control_chars_removed"] = removed

        if cfg.remove_format_chars:
            removed = len(FORMAT_RE.findall(text))
            if removed:
                text = FORMAT_RE.sub("", text)
                file_metrics["format_chars_removed"] = removed

        if cfg.remove_html_comments:
            removed = len(HTML_COMMENT_RE.findall(text))
            if removed:
                text = HTML_COMMENT_RE.sub(" ", text)
                file_metrics["html_comments_removed"] = removed

        if cfg.remove_html_tags:
            removed = len(HTML_TAG_RE.findall(text))
            if removed:
                text = HTML_TAG_RE.sub(" ", text)
                file_metrics["html_tags_removed"] = removed

        if cfg.remove_page_numbers:
            text = self._remove_page_numbers(text, file_metrics)

        if cfg.remove_headers_footers:
            text = self._remove_repeated_lines(text, file_metrics)

        if cfg.remove_markdown_markup:
            text = self._remove_markdown_markup(text, file_metrics)

        if cfg.numeric_cleanup:
            text = self._clean_numbers(text, file_metrics)

        if cfg.lowercase:
            text = self._lowercase_with_protection(text, file_metrics)

        if cfg.remove_stopwords:
            text = self._remove_stopwords(text, file_metrics)

        if cfg.normalize_whitespace:
            text = self._normalize_whitespace(text, file_metrics)

        if cfg.collapse_blank_lines:
            text = BLANK_LINES_RE.sub("\n\n", text)

        text = text.strip()

        if text:
            text += "\n"

        return text

    def _remove_markdown_markup(
        self,
        text: str,
        file_metrics: Dict[str, Any],
    ) -> str:
        headings_removed = len(HEADING_RE.findall(text))
        if headings_removed:
            text = HEADING_RE.sub("", text)
            file_metrics["markdown_headings_removed"] = headings_removed

        blockquotes_removed = len(BLOCKQUOTE_RE.findall(text))
        if blockquotes_removed:
            text = BLOCKQUOTE_RE.sub("", text)
            file_metrics["markdown_blockquotes_removed"] = blockquotes_removed

        lists_removed = len(LIST_RE.findall(text))
        if lists_removed:
            text = LIST_RE.sub("", text)
            file_metrics["markdown_list_markers_removed"] = lists_removed

        hr_removed = len(HR_RE.findall(text))
        if hr_removed:
            text = HR_RE.sub("", text)
            file_metrics["markdown_horizontal_rules_removed"] = hr_removed

        escapes_removed = len(MARKDOWN_ESCAPE_RE.findall(text))
        if escapes_removed:
            file_metrics["markdown_escapes_removed"] = escapes_removed

        text = self._remove_markdown_inline(text)

        return text

    def _remove_markdown_inline(self, text: str) -> str:
        text = INLINE_CODE_RE.sub(lambda m: m.group(1), text)
        text = BOLD_STAR_RE.sub(lambda m: m.group(1), text)
        text = STRIKE_RE.sub(lambda m: m.group(1), text)
        text = MARKDOWN_ESCAPE_RE.sub(lambda m: m.group(1), text)

        return text

    def _clean_numbers(self, text: str, file_metrics: Dict[str, Any]) -> str:
        count = len(DECIMAL_COMMA_RE.findall(text))

        if count:
            text = DECIMAL_COMMA_RE.sub(".", text)
            file_metrics["decimal_commas_normalized"] = count

        return text

    def _lowercase_with_protection(
        self,
        text: str,
        file_metrics: Dict[str, Any],
    ) -> str:
        cfg: CleaningConfig = self.config

        if not cfg.protect_abbreviations:
            return text.lower()

        protected: List[str] = []

        start = "\uE002"
        end = "\uE003"

        def repl(match: re.Match) -> str:
            protected.append(match.group(0))
            return f"{start}{len(protected) - 1}{end}"

        text = ABBREV_RE.sub(repl, text)
        text = text.lower()

        for idx in range(len(protected) - 1, -1, -1):
            marker = f"{start}{idx}{end}"
            text = text.replace(marker, protected[idx])

        if protected:
            file_metrics["abbreviations_protected"] = len(protected)

        return text

    def _load_stopwords(self) -> set:
        if self._stopwords is None:
            words: set = set()

            if nltk_stopwords is not None:
                try:
                    words.update(nltk_stopwords.words("russian"))
                    words.update(nltk_stopwords.words("english"))
                except LookupError:
                    pass

            self._stopwords = words

        return self._stopwords

    def _remove_stopwords(self, text: str, file_metrics: Dict[str, Any]) -> str:
        stop = self._load_stopwords()

        if not stop:
            file_metrics["stopwords_removed"] = 0
            return text

        removed = 0
        lines: List[str] = []

        for line in text.split("\n"):
            if self._is_table_row(line) or PROTECTED_RE.search(line):
                lines.append(line)
                continue

            tokens = line.split(" ")
            kept: List[str] = []

            for token in tokens:
                if not token:
                    kept.append(token)
                    continue

                core = TOKEN_CLEAN_RE.sub("", token)

                if (
                    core
                    and core.lower() in stop
                    and core.isalpha()
                    and not core.isupper()
                    and not PROTECTED_RE.search(token)
                ):
                    removed += 1
                    continue

                kept.append(token)

            lines.append(" ".join(kept))

        if removed:
            file_metrics["stopwords_removed"] = removed

        return "\n".join(lines)

    def _normalize_whitespace(self, text: str, file_metrics: Dict[str, Any]) -> str:
        tabs = text.count("\t")
        nbsp = len(NBSP_RE.findall(text))

        lines: List[str] = []

        for line in text.split("\n"):
            line = line.expandtabs(4)
            line = NBSP_RE.sub(" ", line)
            line = TRAILING_SPACES_RE.sub("", line)
            line = MULTISPACE_RE.sub(" ", line)
            lines.append(line)

        if tabs:
            file_metrics["tabs_normalized"] = tabs

        if nbsp:
            file_metrics["nbsp_normalized"] = nbsp

        return "\n".join(lines)

    def _remove_page_numbers(self, text: str, file_metrics: Dict[str, Any]) -> str:
        removed = len(PAGE_NUMBER_LINE_RE.findall(text))

        if removed:
            text = PAGE_NUMBER_LINE_RE.sub("", text)
            file_metrics["page_numbers_removed"] = removed

        return text

    def _remove_repeated_lines(self, text: str, file_metrics: Dict[str, Any]) -> str:
        cfg: CleaningConfig = self.config

        lines = text.split("\n")
        counts: Dict[str, int] = {}

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            if PROTECTED_RE.search(stripped):
                continue

            if self._is_table_row(stripped):
                continue

            if len(stripped) > cfg.header_max_chars:
                continue

            if len(stripped) < 8 and not any(ch.isdigit() for ch in stripped):
                continue

            counts[stripped] = counts.get(stripped, 0) + 1

        victims = {
            value
            for value, total in counts.items()
            if total >= cfg.header_min_repeats
        }

        if not victims:
            return text

        kept: List[str] = []
        removed = 0

        for line in lines:
            if line.strip() in victims:
                removed += 1
                continue

            kept.append(line)

        if removed:
            file_metrics["header_footer_lines_removed"] = removed

        return "\n".join(kept)

    def finalize_metrics(self, metrics: MetricsCollector) -> None:
        quality_names = [
            "cer",
            "wer",
            "precision",
            "recall",
            "f1",
            "bert_score",
            "llm_as_judge",
            "numbers_saved",
            "formulas_saved",
            "tables_saved",
        ]

        for name in quality_names:
            metrics.set_quality(name, None)

        metrics.meta["quality_note"] = (
            "Качественные метрики пока заглушки. "
            "Их можно реализовать через BaseQualityMetric."
        )

        metrics.meta["expected_quality_metrics"] = quality_names

        metrics.meta["libraries"] = [
            "re",
            "unicodedata",
            "BeautifulSoup",
            "nltk",
        ]

        metrics.meta["note"] = (
            "tabula-py и camelot не используются для .md, "
            "так как они предназначены для извлечения таблиц из PDF."
        )