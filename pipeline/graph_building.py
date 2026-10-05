from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import networkx as nx
from tqdm import tqdm

from .base import BaseStage, MetricsCollector, StageContext


LOGGER = logging.getLogger("graph_building")


def _normalize_label_key(value: str) -> str:
    return str(value).strip().lower().replace("ё", "е")


def _collapse_spaces(value: str) -> str:
    return " ".join(str(value).split())


ELEMENT_UPPER_TO_SYMBOL: Dict[str, str] = {
    "C": "C",
    "MN": "Mn",
    "SI": "Si",
    "S": "S",
    "P": "P",
    "N": "N",
    "H": "H",
    "O": "O",
    "AL": "Al",
    "CA": "Ca",
    "TI": "Ti",
    "V": "V",
    "NB": "Nb",
    "MO": "Mo",
    "CR": "Cr",
    "NI": "Ni",
    "CU": "Cu",
    "W": "W",
    "TA": "Ta",
    "ZR": "Zr",
    "HF": "Hf",
    "B": "B",
}

ELEMENT_LOWER_TO_SYMBOL: Dict[str, str] = {
    "углерод": "C",
    "марганец": "Mn",
    "кремний": "Si",
    "сера": "S",
    "фосфор": "P",
    "азот": "N",
    "водород": "H",
    "кислород": "O",
    "алюминий": "Al",
    "кальций": "Ca",
    "титан": "Ti",
    "ванадий": "V",
    "ниобий": "Nb",
    "молибден": "Mo",
    "хром": "Cr",
    "никель": "Ni",
    "медь": "Cu",
    "вольфрам": "W",
    "тантал": "Ta",
    "цирконий": "Zr",
    "гафний": "Hf",
    "бор": "B",
}

GENERIC_LABELS: Set[str] = {
    "процесс",
    "структура",
    "свойство",
    "параметр",
    "фактор",
    "условие",
    "частицы",
    "выделение",
    "превращение",
    "зона",
    "участок",
    "элемент",
    "материал",
    "характеристика",
    "показатель",
    "режим",
    "вариант",
    "образец",
    "это",
    "такие",
    "данные",
    "результаты",
    "влияние",
    "состояние",
    "область",
    "класс",
    "тип",
    "вид",
    "форма",
    "величина",
    "значение",
}

GENERIC_NORMALIZED: Set[str] = {
    _normalize_label_key(label) for label in GENERIC_LABELS
}

INVALID_LABEL_SUBSTRINGS: Tuple[str, ...] = (
    "[",
    "рис.",
    "табл.",
    "рисунок",
    "таблица",
    "http",
    "https",
    "doi",
)

MEASUREMENT_UNITS: Tuple[str, ...] = (
    "%",
    "°c",
    "°c",
    "мм",
    "см",
    "мкм",
    "нм",
    "мпа",
    "mpa",
    "к/с",
    "k/s",
    "°c/с",
    "°c/c",
    "с/с",
    "ч",
    "мин",
)


def _read_json(path: Union[Path, str]) -> Dict[str, Any]:
    p = Path(path).expanduser()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON config must be an object: {p}")
    return data


def _normalize_type_name(value: Any) -> str:
    return (
        str(value)
        .strip()
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _coerce_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, Iterable):
        parts = list(value)
    else:
        parts = [value]

    result: List[str] = []
    seen: Set[str] = set()

    for part in parts:
        name = _normalize_type_name(part)
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        result.append(name)

    return tuple(result)


@dataclass(frozen=True)
class GraphBuildingConfig:
    encoding: str = "utf-8"

    llm_ollama_model: str = "qwen3.5:9b-q4_K_M"
    llm_ollama_url: str = "http://localhost:11434"
    llm_timeout_seconds: float = 600
    llm_keep_alive: str = "30m"
    llm_context_tokens: int = 8192

    llm_max_new_tokens: int = 768
    llm_max_context_chars: int = 1600
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_repetition_penalty: float = 1.0

    prompt_dir: str = "configs/graph_building"
    extract_prompt_file: str = "extract_metallurgy.txt"
    prompt_file: Optional[str] = None

    output_filename: str = "knowledge_graph"
    output_suffix: str = ".graphml"

    entity_types: Tuple[str, ...] = (
        "МАТЕРИАЛ",
        "ХИМИЧЕСКИЙ_ЭЛЕМЕНТ",
        "СОЕДИНЕНИЕ",
        "МИКРОСТРУКТУРА",
        "ТЕХНОЛОГИЧЕСКИЙ_ПРОЦЕСС",
        "СВОЙСТВО",
        "ПАРАМЕТР",
        "ИЗДЕЛИЕ",
    )

    relation_types: Tuple[str, ...] = (
        "СОДЕРЖИТ",
        "ФОРМИРУЕТ",
        "ПРЕВРАЩАЕТСЯ_В",
        "ТОРМОЗИТ",
        "УСКОРЯЕТ",
        "УЛУЧШАЕТ",
        "УХУДШАЕТ",
        "ВЛИЯЕТ_НА",
        "ПРИМЕНЯЕТСЯ_В",
        "ТРЕБУЕТ",
        "ИЗМЕРЯЕТСЯ",
    )

    default_entity_type: str = "КОНЦЕПЦИЯ"
    default_relation_type: str = "ВЛИЯЕТ_НА"

    max_new_entities: int = 12
    max_new_relations: int = 12
    min_relation_weight: int = 2
    max_quote_chars: int = 100
    max_label_chars: int = 80

    strict_types: bool = True
    require_quote: bool = True
    drop_unresolved_relations: bool = True
    deduplicate_relations: bool = True
    remove_isolates: bool = True
    store_edge_quote: bool = False
    filter_labels: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_types", _coerce_tuple(self.entity_types))
        object.__setattr__(self, "relation_types", _coerce_tuple(self.relation_types))

        if not self.entity_types:
            raise ValueError("entity_types must not be empty")
        if not self.relation_types:
            raise ValueError("relation_types must not be empty")
        if self.llm_max_new_tokens <= 0:
            raise ValueError("llm_max_new_tokens must be positive")
        if self.llm_context_tokens <= self.llm_max_new_tokens:
            raise ValueError("llm_context_tokens must exceed llm_max_new_tokens")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be positive")
        if not self.llm_ollama_model.strip():
            raise ValueError("llm_ollama_model must not be empty")
        if self.llm_max_context_chars <= 0:
            raise ValueError("llm_max_context_chars must be positive")
        if self.max_new_entities <= 0:
            raise ValueError("max_new_entities must be positive")
        if self.max_new_relations <= 0:
            raise ValueError("max_new_relations must be positive")
        if self.min_relation_weight < 1:
            raise ValueError("min_relation_weight must be >= 1")
        if self.max_quote_chars <= 0:
            raise ValueError("max_quote_chars must be positive")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["entity_types"] = list(self.entity_types)
        data["relation_types"] = list(self.relation_types)
        return data

    @classmethod
    def from_json(
        cls,
        path: Union[Path, str],
        *,
        ontology_path: Optional[Union[Path, str]] = None,
        prompt_path: Optional[Union[Path, str]] = None,
        llm_config_path: Optional[Union[Path, str]] = None,
        **overrides: Any,
    ) -> "GraphBuildingConfig":
        """
        Loads base config and optionally merges:
          - ontology config: entity_types, relation_types, defaults;
          - prompt file or prompt directory;
          - llm config: keys starting with llm_;
          - direct keyword overrides.
        """
        data = _read_json(path)
        known_fields = {field.name for field in fields(cls)}

        if llm_config_path is not None:
            llm_data = _read_json(llm_config_path)
            data.update(
                {
                    key: value
                    for key, value in llm_data.items()
                    if key.startswith("llm_") and key in known_fields
                }
            )

        if ontology_path is not None:
            ontology_data = _read_json(ontology_path)
            for key in (
                "entity_types",
                "relation_types",
                "default_entity_type",
                "default_relation_type",
            ):
                if key in ontology_data and key in known_fields:
                    data[key] = ontology_data[key]

        if prompt_path is not None:
            p = Path(prompt_path).expanduser()
            if p.is_dir():
                data["prompt_dir"] = str(p)
            else:
                data["prompt_file"] = str(p)

        data.update(
            {
                key: value
                for key, value in overrides.items()
                if key in known_fields
            }
        )

        clean_data = {
            key: value
            for key, value in data.items()
            if key in known_fields
        }

        return cls(**clean_data)


class GraphBuildingStage(BaseStage):
    name = "graph_building"

    def __init__(self, config: Optional[GraphBuildingConfig] = None) -> None:
        super().__init__(config or GraphBuildingConfig())

        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._prompt_template: Optional[str] = None

        self._aliases: Dict[str, str] = {}
        self._label_to_nodes: Dict[str, Set[str]] = {}
        self._node_ids_by_label_type: Dict[Tuple[str, str], str] = {}
        self._seen_edges: Dict[Tuple[str, str, str], Tuple[str, str, Any]] = {}

        self._type_priority: Dict[str, int] = {}
        for idx, entity_type in enumerate(self.config.entity_types):
            self._type_priority[entity_type] = idx

        self._type_priority[self.config.default_entity_type] = (
            max(self._type_priority.values(), default=-1) + 1
        )

    def validate(self, ctx: StageContext) -> None:
        super().validate(ctx)
        self._load_prompt_template(ctx)

        if not self._prompt_template:
            raise FileNotFoundError(
                "Graph extraction prompt was not found. "
                "Check prompt_file / prompt_dir / extract_prompt_file."
            )

    def list_input_files(self, ctx: StageContext) -> List[Path]:
        return sorted(
            path for path in ctx.input_dir.glob("*.vectors.json")
            if not path.name.endswith(".tables.vectors.json")
        )

    def run(self, ctx: StageContext) -> Any:
        result = super().run(ctx)

        if result.success:
            self._save_graphml(ctx.output_dir)

        return result

    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        metrics: MetricsCollector,
    ) -> Path:
        cfg: GraphBuildingConfig = self.config
        record = json.loads(input_path.read_text(encoding=cfg.encoding))

        chunks = record.get("chunks", [])
        if not isinstance(chunks, list):
            return input_path

        for chunk in tqdm(
            chunks,
            desc=f"[graph] {input_path.name.split('.')[0]}",
            unit="chunk",
            position=1,
            leave=False,
            dynamic_ncols=True,
        ):
            if not isinstance(chunk, dict):
                continue

            chunk_text = (
                chunk.get("dense_text", "")
                or chunk.get("text", "")
                or ""
            ).strip()

            if not chunk_text:
                continue

            llm_chunk_text = chunk_text[: cfg.llm_max_context_chars]

            llm_entities, llm_relations = self._extract_with_llm(
                chunk_text=llm_chunk_text,
            )

            metrics.inc("llm_entities_returned", len(llm_entities))
            metrics.inc("llm_relations_returned", len(llm_relations))

            for entity in llm_entities:
                prepared = self._prepare_entity(
                    raw_type=entity.get("type", ""),
                    raw_label=entity.get("name", ""),
                )

                if prepared is None:
                    metrics.inc("dropped_invalid_entity")
                    continue

                node_id = self._register_node(*prepared)

                if node_id is None:
                    metrics.inc("dropped_invalid_entity")
                else:
                    metrics.inc("entities_registered")

            normalized_chunk = self._normalize_quote(llm_chunk_text)

            for relation in llm_relations:
                self._add_relation(
                    relation=relation,
                    normalized_chunk=normalized_chunk,
                    metrics=metrics,
                )

            metrics.inc("chunks_processed")

        return input_path

    def finalize_metrics(self, metrics: MetricsCollector) -> None:
        metrics.set_gauge("total_nodes", self._graph.number_of_nodes())
        metrics.set_gauge("total_edges", self._graph.number_of_edges())

        node_types: Dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            node_type = str(data.get("node_type", "Unknown"))
            node_types[node_type] = node_types.get(node_type, 0) + 1

        edge_types: Dict[str, int] = {}
        for _, _, data in self._graph.edges(data=True):
            edge_type = str(data.get("relation", "Unknown"))
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        metrics.set_gauge("nodes_by_type", node_types)
        metrics.set_gauge("edges_by_type", edge_types)

        chunks_processed = metrics.counters.get("chunks_processed", 0)
        relations_added = metrics.counters.get("relations_added", 0)
        relations_returned = metrics.counters.get("llm_relations_returned", 0)
        entities_returned = metrics.counters.get("llm_entities_returned", 0)

        if chunks_processed:
            metrics.set_quality(
                "relations_per_chunk",
                relations_added / chunks_processed,
            )
            metrics.set_quality(
                "entities_per_chunk",
                entities_returned / chunks_processed,
            )
        else:
            metrics.set_quality("relations_per_chunk", None)
            metrics.set_quality("entities_per_chunk", None)

        if relations_returned:
            metrics.set_quality(
                "relation_acceptance_rate",
                relations_added / relations_returned,
            )
        else:
            metrics.set_quality("relation_acceptance_rate", None)

        dropped_total = (
            metrics.counters.get("dropped_invalid_entity", 0)
            + metrics.counters.get("dropped_invalid_relation_type", 0)
            + metrics.counters.get("dropped_low_weight_relation", 0)
            + metrics.counters.get("dropped_missing_quote", 0)
            + metrics.counters.get("dropped_long_quote", 0)
            + metrics.counters.get("dropped_unsupported_quote", 0)
            + metrics.counters.get("dropped_unresolved_source", 0)
            + metrics.counters.get("dropped_unresolved_target", 0)
            + metrics.counters.get("dropped_self_relation", 0)
            + metrics.counters.get("dropped_duplicate_relation", 0)
        )

        metrics.set_gauge("dropped_total", dropped_total)

    def _load_prompt_template(self, ctx: StageContext) -> None:
        cfg: GraphBuildingConfig = self.config
        candidates: List[Path] = []

        if cfg.prompt_file:
            candidates.append(Path(cfg.prompt_file).expanduser())

        prompt_dir = Path(cfg.prompt_dir).expanduser()

        if prompt_dir.is_absolute():
            prompt_dirs = [prompt_dir]
        else:
            prompt_dirs = [
                Path.cwd() / prompt_dir,
                ctx.input_dir / prompt_dir,
                ctx.output_dir / prompt_dir,
            ]

        for directory in prompt_dirs:
            candidates.append(directory / cfg.extract_prompt_file)

        for candidate in candidates:
            if candidate.exists():
                self._prompt_template = candidate.read_text(
                    encoding=cfg.encoding
                )
                LOGGER.info("Graph extraction prompt loaded from %s", candidate)
                return

        LOGGER.error(
            "Graph extraction prompt was not found. Tried: %s",
            [str(p) for p in candidates],
        )
        self._prompt_template = None

    def _extract_with_llm(
        self,
        chunk_text: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self._prompt_template:
            return [], []

        prompt = self._render_prompt(chunk_text=chunk_text)
        raw_output = self._generate(prompt)
        return self._parse_llm_output(raw_output)

    def _render_prompt(self, chunk_text: str) -> str:
        cfg: GraphBuildingConfig = self.config

        replacements = {
            "{entity_types}": ", ".join(cfg.entity_types),
            "{relation_types}": ", ".join(cfg.relation_types),
            "{chunk_text}": chunk_text,
            "{max_new_entities}": str(cfg.max_new_entities),
            "{max_new_relations}": str(cfg.max_new_relations),
            "{min_relation_weight}": str(cfg.min_relation_weight),
            "{max_quote_chars}": str(cfg.max_quote_chars),
        }

        prompt = self._prompt_template or ""

        for key, value in replacements.items():
            prompt = prompt.replace(key, value)

        return prompt

    def _parse_llm_output(
        self,
        text: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        cfg: GraphBuildingConfig = self.config

        entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []

        def handle_item(item: Any) -> None:
            if not isinstance(item, list) or not item:
                return

            if isinstance(item[0], list):
                for child in item:
                    handle_item(child)
                return

            kind = str(item[0]).strip().upper()

            if kind == "E" and len(item) >= 2:
                name = str(item[1]).strip()
                entity_type = _normalize_type_name(item[2]) if len(item) > 2 else ""

                if not name:
                    return

                entities.append(
                    {
                        "name": name,
                        "type": entity_type,
                    }
                )

            elif kind == "R" and len(item) >= 4:
                source = str(item[1]).strip()
                target = str(item[2]).strip()
                relation_type = _normalize_type_name(item[3])
                quote = str(item[4]).strip() if len(item) > 4 else ""

                if not source or not target:
                    return

                try:
                    weight = (
                        int(item[5])
                        if len(item) > 5
                        else cfg.min_relation_weight
                    )
                except (TypeError, ValueError):
                    weight = cfg.min_relation_weight

                if weight < cfg.min_relation_weight:
                    return

                relations.append(
                    {
                        "source": source,
                        "target": target,
                        "type": relation_type,
                        "quote": quote,
                        "weight": weight,
                    }
                )

        # Accept JSONL and wrapped arrays. If an outer array is incomplete,
        # recover only its individually complete JSON records, never repair them.
        decoder = json.JSONDecoder()
        position = 0
        while position < len(text):
            start = text.find("[", position)
            if start < 0:
                break
            try:
                item, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                position = start + 1
                continue
            handle_item(item)
            position = end

        return (
            entities[: cfg.max_new_entities],
            relations[: cfg.max_new_relations],
        )

    def _prepare_entity(
        self,
        raw_type: Any,
        raw_label: Any,
    ) -> Optional[Tuple[str, str]]:
        cfg: GraphBuildingConfig = self.config

        entity_type = _normalize_type_name(raw_type)

        if entity_type not in cfg.entity_types:
            if cfg.strict_types:
                return None
            entity_type = cfg.default_entity_type

        label = self._clean_label(raw_label)
        label = self._canonicalize_label(label)

        if not label:
            return None

        if cfg.filter_labels and not self._is_valid_label(label):
            return None

        return entity_type, label

    def _register_node(
        self,
        entity_type: str,
        label: str,
    ) -> Optional[str]:
        cfg: GraphBuildingConfig = self.config

        clean_label = self._clean_label(label)
        clean_label = self._canonicalize_label(clean_label)

        if not clean_label:
            return None

        if cfg.filter_labels and not self._is_valid_label(clean_label):
            return None

        normalized_type = _normalize_type_name(entity_type)
        if not normalized_type:
            normalized_type = cfg.default_entity_type

        key = (normalized_type, clean_label)
        node_id = self._node_ids_by_label_type.get(key)

        if node_id is None:
            hash_source = f"{normalized_type}::{clean_label}".encode("utf-8")
            node_id = "N" + hashlib.sha1(hash_source).hexdigest()[:16]

            self._graph.add_node(
                node_id,
                node_type=normalized_type,
                label=clean_label,
            )

            self._node_ids_by_label_type[key] = node_id

        normalized_label = _normalize_label_key(clean_label)

        self._label_to_nodes.setdefault(normalized_label, set()).add(node_id)

        current_alias = self._aliases.get(normalized_label)
        if current_alias is None:
            self._aliases[normalized_label] = node_id
        else:
            current_type = str(
                self._graph.nodes[current_alias].get("node_type", "")
            )
            if self._priority(normalized_type) < self._priority(current_type):
                self._aliases[normalized_label] = node_id

        return node_id

    def _resolve_node(self, name: str) -> Optional[str]:
        normalized = _normalize_label_key(name)

        node_id = self._aliases.get(normalized)
        if node_id is not None:
            return node_id

        candidates = self._label_to_nodes.get(normalized)
        if candidates:
            return min(
                candidates,
                key=lambda nid: self._priority(
                    str(self._graph.nodes[nid].get("node_type", ""))
                ),
            )

        return None

    def _priority(self, entity_type: str) -> int:
        return self._type_priority.get(entity_type, 9999)

    def _add_relation(
        self,
        relation: Dict[str, Any],
        normalized_chunk: str,
        metrics: MetricsCollector,
    ) -> bool:
        cfg: GraphBuildingConfig = self.config

        relation_type = _normalize_type_name(relation.get("type", ""))

        if relation_type not in cfg.relation_types:
            if cfg.strict_types:
                metrics.inc("dropped_invalid_relation_type")
                return False
            relation_type = cfg.default_relation_type

        try:
            weight = int(relation.get("weight", cfg.min_relation_weight))
        except (TypeError, ValueError):
            weight = cfg.min_relation_weight

        if weight < cfg.min_relation_weight:
            metrics.inc("dropped_low_weight_relation")
            return False

        quote = str(relation.get("quote", "")).strip()

        if cfg.require_quote:
            if not quote:
                metrics.inc("dropped_missing_quote")
                return False

            if len(quote) > cfg.max_quote_chars * 2:
                metrics.inc("dropped_long_quote")
                return False

            if not self._quote_is_supported(quote, normalized_chunk):
                metrics.inc("dropped_unsupported_quote")
                return False

        source_id = self._resolve_node(str(relation.get("source", "")))
        target_id = self._resolve_node(str(relation.get("target", "")))

        if source_id is None:
            if cfg.drop_unresolved_relations:
                metrics.inc("dropped_unresolved_source")
                return False

            prepared = self._prepare_entity(
                cfg.default_entity_type,
                relation.get("source", ""),
            )

            if prepared is None:
                metrics.inc("dropped_unresolved_source")
                return False

            source_id = self._register_node(*prepared)

        if target_id is None:
            if cfg.drop_unresolved_relations:
                metrics.inc("dropped_unresolved_target")
                return False

            prepared = self._prepare_entity(
                cfg.default_entity_type,
                relation.get("target", ""),
            )

            if prepared is None:
                metrics.inc("dropped_unresolved_target")
                return False

            target_id = self._register_node(*prepared)

        if source_id is None or target_id is None:
            metrics.inc("dropped_unresolved_relation")
            return False

        if source_id == target_id:
            metrics.inc("dropped_self_relation")
            return False

        edge_id = (source_id, target_id, relation_type)

        if cfg.deduplicate_relations and edge_id in self._seen_edges:
            u, v, edge_key = self._seen_edges[edge_id]
            edge_data = self._graph.edges[u, v, edge_key]

            try:
                old_weight = int(edge_data.get("weight", 0))
            except (TypeError, ValueError):
                old_weight = 0

            if weight > old_weight:
                edge_data["weight"] = str(weight)

                if cfg.store_edge_quote:
                    edge_data["quote"] = quote[: cfg.max_quote_chars]

                metrics.inc("relations_updated")

            metrics.inc("dropped_duplicate_relation")
            return False

        edge_attrs: Dict[str, Any] = {
            "relation": relation_type,
            "weight": str(weight),
        }

        if cfg.store_edge_quote:
            edge_attrs["quote"] = quote[: cfg.max_quote_chars]

        edge_key = self._graph.add_edge(source_id, target_id, **edge_attrs)
        self._seen_edges[edge_id] = (source_id, target_id, edge_key)

        metrics.inc("relations_added")
        return True

    def _clean_label(self, value: Any) -> str:
        cfg: GraphBuildingConfig = self.config

        text = str(value).strip()
        text = text.strip('"').strip("'")
        text = _collapse_spaces(text)

        if len(text) > cfg.max_label_chars:
            text = text[: cfg.max_label_chars].rstrip()

        return text

    def _canonicalize_label(self, label: str) -> str:
        raw = label.strip()

        upper = raw.upper()
        if upper in ELEMENT_UPPER_TO_SYMBOL:
            return ELEMENT_UPPER_TO_SYMBOL[upper]

        lower = raw.lower().replace("ё", "е")
        if lower in ELEMENT_LOWER_TO_SYMBOL:
            return ELEMENT_LOWER_TO_SYMBOL[lower]

        return raw

    def _is_valid_label(self, label: str) -> bool:
        text = str(label).strip()

        if not text:
            return False

        normalized = _normalize_label_key(text)

        if normalized in GENERIC_NORMALIZED:
            return False

        lowered = text.lower()

        if any(substring in lowered for substring in INVALID_LABEL_SUBSTRINGS):
            return False

        has_digit = any(ch.isdigit() for ch in text)
        has_alpha = any(ch.isalpha() for ch in text)

        if has_digit and not has_alpha:
            return False

        if has_digit and any(unit in lowered for unit in MEASUREMENT_UNITS):
            return False

        if len(text.split()) > 8:
            return False

        return True

    def _normalize_quote(self, value: str) -> str:
        text = str(value).lower().replace("ё", "е")

        text = text.replace("—", "-")
        text = text.replace("–", "-")
        text = text.replace("−", "-")

        return _collapse_spaces(text)

    def _quote_is_supported(self, quote: str, normalized_chunk: str) -> bool:
        if not quote:
            return False

        normalized_quote = self._normalize_quote(quote)
        if not normalized_quote:
            return False

        return normalized_quote in normalized_chunk

    def _generate(self, prompt: str) -> str:
        cfg: GraphBuildingConfig = self.config
        payload = {
            "model": cfg.llm_ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": cfg.llm_keep_alive,
            "options": {
                "num_ctx": cfg.llm_context_tokens,
                "num_predict": cfg.llm_max_new_tokens,
                "temperature": cfg.llm_temperature,
                "top_p": cfg.llm_top_p,
                "repeat_penalty": cfg.llm_repetition_penalty,
            },
        }
        request = Request(
            cfg.llm_ollama_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=cfg.llm_timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code}: {detail}. "
                f"Check the model with: ollama pull {cfg.llm_ollama_model}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Ollama request failed at {cfg.llm_ollama_url}: {exc}. "
                "Start Ollama; if it is already running, check the request timeout."
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError("Ollama returned an invalid response")
        if result.get("error"):
            raise RuntimeError(f"Ollama: {result['error']}")
        if not result.get("done"):
            raise RuntimeError("Ollama returned an incomplete response")

        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("Ollama response is missing message.content")
        if result.get("done_reason") == "length":
            LOGGER.warning(
                "Ollama reached llm_max_new_tokens=%s; JSONL may be incomplete",
                cfg.llm_max_new_tokens,
            )
        return content

    def _save_graphml(self, output_dir: Path) -> None:
        cfg: GraphBuildingConfig = self.config

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{cfg.output_filename}{cfg.output_suffix}"

        if cfg.remove_isolates:
            isolates = list(nx.isolates(self._graph))
            self._graph.remove_nodes_from(isolates)

        for node, data in list(self._graph.nodes(data=True)):
            for key in list(data.keys()):
                self._graph.nodes[node][key] = str(data[key])

        for u, v, key, data in list(self._graph.edges(keys=True, data=True)):
            for attr_key in list(data.keys()):
                self._graph.edges[u, v, key][attr_key] = str(data[attr_key])

        nx.write_graphml(self._graph, output_path)

        LOGGER.info(
            "Graph saved: %s (%s nodes, %s edges)",
            output_path,
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )
