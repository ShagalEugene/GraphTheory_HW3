from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

DISPLAY_MATH_RE = re.compile(
    r"(?ms)\$\$.*?\$\$|\\\[.*?\\\]"
)

INLINE_MATH_RE = re.compile(
    r"(?<!\$)\$(?!\s)[^$\n]+?(?<!\s)\$(?!\$)|\\\(.*?\\\)"
)

PROTECTED_RE = re.compile(r"[\uE000-\uE003]")

UNIT_VOCAB: Dict[str, str] = {
    "Н/мм²": "N/mm**2",
    "кг/м³": "kg/m**3",
    "г/см³": "g/cm**3",
    "т/м³": "tonne/m**3",
    "Н/мм2": "N/mm**2",
    "кг/м3": "kg/m**3",
    "г/см3": "g/cm**3",
    "мм2": "mm**2",
    "см2": "cm**2",
    "м2": "m**2",
    "мм3": "mm**3",
    "см3": "cm**3",
    "м3": "m**3",
    "км/ч": "km/h",
    "м/с": "m/s",
    "К/с": "K/s",
    "с/мм": "s/mm",
    "об/мин": "rpm",
    "Н·м": "N*m",
    "мм²": "mm**2",
    "см²": "cm**2",
    "м²": "m**2",
    "мм³": "mm**3",
    "см³": "cm**3",
    "м³": "m**3",
    "мкм": "um",
    "мм": "mm",
    "см": "cm",
    "дм": "dm",
    "км": "km",
    "мг": "mg",
    "кг": "kg",
    "мс": "ms",
    "мин": "min",
    "сут": "day",
    "кПа": "kPa",
    "МПа": "MPa",
    "ГПа": "GPa",
    "кН": "kN",
    "МН": "MN",
    "кВ": "kV",
    "мА": "mA",
    "кГц": "kHz",
    "МГц": "MHz",
    "кВт": "kW",
    "МВт": "MW",
    "кДж": "kJ",
    "мл": "ml",
    "°C": "degC",
    "Па": "Pa",
    "Н": "N",
    "м": "m",
    "с": "s",
    "ч": "h",
    "т": "tonne",
    "В": "V",
    "А": "A",
    "Ом": "ohm",
    "Вт": "W",
    "Дж": "J",
    "Гц": "Hz",
    "л": "l",
    "К": "K",
    "рад": "rad",
    "K": "K",
    "m": "m",
    "s": "s",
    "kg": "kg",
    "Pa": "Pa",
    "MPa": "MPa",
    "m/s": "m/s",
    "K/s": "K/s",
    "m**2": "m**2",
    "m**3": "m**3",
    "kg/m**3": "kg/m**3",
    "kg*m**2/s**2": "kg*m**2/s**2",
    "kg/(m*s**2)": "kg/(m*s**2)",
    "kg/m/s**2": "kg/m/s**2",
}

UNIT_ALTERNATION = "|".join(
    sorted(
        (re.escape(unit) for unit in UNIT_VOCAB),
        key=len,
        reverse=True,
    )
)

NUMUNIT_RE = re.compile(
    rf"(?<![0-9A-Za-zА-Яа-яЁё.,\-–—])(-?\d+(?:[.,]\d+)?)\s?({UNIT_ALTERNATION})(?![0-9A-Za-zА-Яа-яЁё])"
)

class TextProtector:
    START = "\uE000"
    END = "\uE001"

    def __init__(self) -> None:
        self.blocks: List[Tuple[str, str]] = []

    def add(self, kind: str, content: str) -> str:
        marker = f"{self.START}{len(self.blocks)}{self.END}"
        self.blocks.append((kind, content))
        return marker

    def protect(
        self,
        text: str,
        pattern: re.Pattern,
        kind: str,
        block: bool = False,
        transform: Optional[Callable[[str], str]] = None,
    ) -> Tuple[str, int]:
        count = 0

        def repl(match: re.Match) -> str:
            nonlocal count
            count += 1

            content = match.group(0)

            if transform is not None:
                try:
                    content = transform(content)
                except Exception:
                    content = match.group(0)

            marker = self.add(kind, content)

            if block:
                return f"\n{marker}\n"

            return marker

        text = pattern.sub(repl, text)

        return text, count

    def restore(self, text: str) -> str:
        for idx in range(len(self.blocks) - 1, -1, -1):
            marker = f"{self.START}{idx}{self.END}"
            _, content = self.blocks[idx]
            text = text.replace(marker, content)
        return text