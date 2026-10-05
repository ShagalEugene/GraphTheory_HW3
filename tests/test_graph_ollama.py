import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import networkx as nx

from pipeline.base import StageContext
from pipeline.graph_building import GraphBuildingConfig, GraphBuildingStage


@contextmanager
def ollama_server(response, status=200):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, json.loads(body)))
            encoded = json.dumps(response).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class GraphOllamaTests(unittest.TestCase):
    def test_wrapped_and_incomplete_outer_arrays_preserve_complete_records(self):
        stage = GraphBuildingStage()
        entity = ["E", "Nb", "ХИМИЧЕСКИЙ_ЭЛЕМЕНТ"]
        relation = ["R", "Nb", "сталь", "СОДЕРЖИТ", "сталь содержит ниобий", 3]
        wrapped = json.dumps([entity, [relation]], ensure_ascii=False)
        for answer in (wrapped, wrapped[:-1]):
            with self.subTest(answer=answer):
                entities, relations = stage._parse_llm_output(answer)
                self.assertEqual(len(entities), 1)
                self.assertEqual(len(relations), 1)
                self.assertEqual(entities[0]["name"], "Nb")
        entities, relations = stage._parse_llm_output(
            json.dumps(entity) + '\n["R","Nb","unfinished'
        )
        self.assertEqual(len(entities), 1)
        self.assertEqual(relations, [])

    def test_request_preserves_unicode_and_disables_thinking(self):
        answer = '["E","ниобий","ХИМИЧЕСКИЙ_ЭЛЕМЕНТ"]'
        with ollama_server({"done": True, "message": {"content": answer}}) as (url, calls):
            stage = GraphBuildingStage(GraphBuildingConfig(llm_ollama_url=url))
            self.assertEqual(stage._generate("ниобий тормозит рекристаллизацию"), answer)
        endpoint, request = calls[0]
        self.assertEqual(endpoint, "/api/chat")
        self.assertEqual(request["messages"][0]["content"], "ниобий тормозит рекристаллизацию")
        self.assertEqual(request["model"], "qwen3.5:9b-q4_K_M")
        self.assertFalse(request["think"])
        self.assertFalse(request["stream"])
        self.assertEqual(request["options"]["num_ctx"], 8192)
        self.assertNotIn("format", request)  # Extraction expects JSONL, not one JSON object.

    def test_missing_model_has_actionable_error(self):
        with ollama_server({"error": "model not found"}, status=404) as (url, _):
            stage = GraphBuildingStage(GraphBuildingConfig(llm_ollama_url=url))
            with self.assertRaisesRegex(RuntimeError, "ollama pull qwen3.5"):
                stage._generate("text")

    def test_empty_extraction_is_valid_but_missing_content_is_not(self):
        with ollama_server({"done": True, "message": {"content": ""}}) as (url, _):
            stage = GraphBuildingStage(GraphBuildingConfig(llm_ollama_url=url))
            self.assertEqual(stage._generate("no relationships"), "")
        with ollama_server({"done": True, "message": {}}) as (url, _):
            stage = GraphBuildingStage(GraphBuildingConfig(llm_ollama_url=url))
            with self.assertRaisesRegex(RuntimeError, "message.content"):
                stage._generate("text")

    def test_graph_stage_exports_supported_relation_and_skips_table_artifacts(self):
        quote = "ниобий тормозит рекристаллизацию"
        answer = "\n".join([
            json.dumps(["E", "Nb", "ХИМИЧЕСКИЙ_ЭЛЕМЕНТ"], ensure_ascii=False),
            json.dumps(["E", "рекристаллизация", "ТЕХНОЛОГИЧЕСКИЙ_ПРОЦЕСС"], ensure_ascii=False),
            json.dumps(["R", "Nb", "рекристаллизация", "ТОРМОЗИТ", quote, 3], ensure_ascii=False),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "vectors"
            source.mkdir()
            (source / "sample.vectors.json").write_text(
                json.dumps({"chunks": [{"dense_text": quote}]}), encoding="utf-8",
            )
            (source / "sample.tables.vectors.json").write_text(
                json.dumps({"tables": []}), encoding="utf-8",
            )
            prompt = root / "extract.txt"
            prompt.write_text("{chunk_text}", encoding="utf-8")
            with ollama_server({"done": True, "message": {"content": answer}}) as (url, calls):
                stage = GraphBuildingStage(GraphBuildingConfig(
                    llm_ollama_url=url, prompt_file=str(prompt),
                ))
                result = stage.run(StageContext(input_dir=source, output_dir=root / "output"))
            self.assertTrue(result.success, result.to_dict())
            self.assertEqual(len(calls), 1)
            graph = nx.read_graphml(root / "output" / "knowledge_graph.graphml")
            self.assertEqual(graph.number_of_nodes(), 2)
            self.assertEqual(graph.number_of_edges(), 1)
            self.assertEqual(next(iter(graph.edges(data=True)))[2]["relation"], "ТОРМОЗИТ")
            metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["counters"]["chunks_processed"], 1)

    def test_api_error_marks_stage_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "vectors"
            source.mkdir()
            (source / "sample.vectors.json").write_text(
                json.dumps({"chunks": [{"text": "text"}]}), encoding="utf-8",
            )
            prompt = root / "extract.txt"
            prompt.write_text("{chunk_text}", encoding="utf-8")
            with ollama_server({"error": "model not found"}, status=404) as (url, _):
                stage = GraphBuildingStage(GraphBuildingConfig(
                    llm_ollama_url=url, prompt_file=str(prompt),
                ))
                with self.assertLogs("pipeline", level="ERROR"):
                    result = stage.run(StageContext(input_dir=source, output_dir=root / "output"))
            self.assertFalse(result.success)
            self.assertIn("model not found", result.warnings[0])
            self.assertFalse((root / "output" / "knowledge_graph.graphml").exists())


if __name__ == "__main__":
    unittest.main()
