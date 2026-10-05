from pathlib import Path
from pipeline import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def run_stage(stage, input_dir: Path, output_root: Path):
    ctx = StageContext(
        input_dir=input_dir,
        output_dir=output_root,
    )

    result = stage.run(ctx)

    if not result.success:
        raise RuntimeError(result.to_dict())

    return result.output_dir


def main():
    project_root = Path(__file__).resolve().parent
    input_dir = project_root / "input"
    output_root = project_root / "output"

    abbreviations ={
        "ррт": "ppm",
        "Interstitial-Free стали": "IF-стали",
        "Interstitial-Free": "IF",
        "IF стали": "IF-стали",
        "Transformation Induced Plasticity стали": "TRIP-стали",
        "Transformation Induced Plasticity": "TRIP",
        "TRIP стали": "TRIP-стали",
        "Drop Weight Tear Test": "DWTT",
        "Lueders elongation": "L.El",
        "L. El": "L.El",
        "Product Specification Level": "PSL",
        "Welding Crack Sensitivity": "PCM",
        "Crack Parameter": "PCM",
    }

    terms = (
        "ТМКП", "ТМО", "ОШЗ", "МА", "ИПГ", "DWTT", "L.El",
        "IF", "IF-стали", "TRIP", "TRIP-стали",
        "ОАО", "РАО", "МК", "ВНИИГАЗ", "ВНИИСТ", "ХТЗ",
        "ЦНИИчермет", "ИТЦ",
        "ГОСТ", "API", "ASTM", "PSL", "ТТ",
        "К60", "К65", "X70", "X80", "CE", "PCM", "KCV", "KCU",
        "HV", "Sv", "T5", "T95", "Ar3", "Ac1", "Ac3", "ppm", "Me/C",
        "C", "Mn", "Si", "S", "P", "N", "H", "O", "Al", "Ca", "Ti",
        "V", "Nb", "Mo", "Cr", "Ni", "Cu", "W", "Ta", "Zr", "Hf", "B",
    )

    current_input = input_dir

    model_path = project_root / "models" / "multilingual-e5-small"

    graph_config = GraphBuildingConfig.from_json(
        project_root / "configs" / "config.json",
        ontology_path=(
            project_root
            / "configs"
            / "ontology.json"
        ),
        prompt_path=(
            project_root
            / "configs"
            / "extract.txt"
        ),
        llm_config_path=(
            project_root
            / "configs"
            / "llm.json"
        ),
    )

    stages = [
        ClearingStage(
            CleaningConfig(
                lowercase=False,
                remove_stopwords=False,
                numeric_cleanup=False,
                protect_math=True,
                remove_images=True,
                remove_markdown_links=True,
                remove_markdown_markup=True,
                convert_html_tables_to_markdown=True,
                protect_markdown_tables=True,
            )
        ),
        NormalizationStage(
            NormalizationConfig(
                normalize_units=True,
                normalize_dates=True,
                normalize_numbers=True,
                normalize_formulas=True,
                normalize_tables=True,
                lemmatization="lemmatize",
                abbreviations=abbreviations,
                terms=terms
            )
        ),
        ChunkingStage(
            ChunkingConfig(
                chunk_size_tokens=800,
                chunk_overlap_percent=0.15,
            )
        ),
        TokenizationStage(
            TokenizationConfig(
                mode="hybrid",
                formula_token_mode="single",
                terms=terms,
            )
        ),
        VectorizationStage(
            VectorizationConfig(
                dense_model_path=str(model_path),
                dense_model_repo="intfloat/multilingual-e5-small",
                dense_model_auto_download=True,
                dense_device="cpu",
                dense_max_seq_length=512,
                dense_passage_prefix="passage: ",
                sparse_method="tfidf",
                token_source="cl100k_base",
                use_normalized_tables=True,
                include_table_tokens_in_chunk_sparse=True,
                table_text_in_chunk_dense=True,
                formula_mode="ast",
                table_mode="header_type_rows",
                vector_store="qdrant",
            )
        ),
        GraphBuildingStage(graph_config),
    ]

    for stage in stages:
        current_input = run_stage(
            stage=stage,
            input_dir=current_input,
            output_root=output_root,
        )

    print(f"Graph: {output_root / 'knowledge_graph.graphml'}")


if __name__ == "__main__":
    main()
