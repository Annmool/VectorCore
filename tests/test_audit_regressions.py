import json
from unittest.mock import patch

import numpy as np

from vectordb.core.collection import Collection, InvalidMetadataFilter, compile_metadata_filter
from vectordb.core.flat_index import FlatIndex
from vectordb.core.hnsw_index import HNSWIndex
from vectordb.core.ivf_index import IVFIndex
from vectordb.core.storage import StorageEngine
from vectordb.evaluation.metrics import average_precision, ndcg_at_k, recall_at_k
from vectordb.retrieval.bm25 import BM25Index
from vectordb.retrieval.hybrid import HybridRetriever


def test_hybrid_filters_both_candidate_pools(tmp_path):
    class Embedder:
        def embed_query(self, query):
            return [1, 0]

    storage = StorageEngine(str(tmp_path))
    collection = Collection("filter", 2, "flat", storage_engine=storage)
    ids = ["private", "public"]
    texts = ["falcon secret", "falcon memo"]
    metadata = [{"team": team, "text": text} for team, text in zip(ids, texts)]
    collection.insert_batch(ids, [[1, 0], [0, 1]], metadata)
    bm25 = BM25Index()
    bm25.add_documents(ids, texts, metadata)
    retriever = HybridRetriever(collection, bm25, Embedder())
    assert [row["id"] for row in retriever.search("falcon", filter={"team": "public"})] == ["public"]

    try:
        compile_metadata_filter({"year": {"$gte_typo": 2024}})
    except InvalidMetadataFilter:
        pass
    else:
        raise AssertionError("Unknown filter operator was accepted")


def test_failed_snapshot_publish_preserves_last_commit(tmp_path):
    storage = StorageEngine(str(tmp_path))
    collection = Collection("atomic", 2, "flat", storage_engine=storage)
    collection.insert("old", [1, 0])
    collection.save()
    collection.insert("new", [0, 1])

    with patch("vectordb.core.storage.os.replace", side_effect=OSError("injected")):
        try:
            collection.save()
        except OSError:
            pass
        else:
            raise AssertionError("Save should report publish failure")

    restored = Collection.load("atomic", storage)
    assert restored.index.ids == ["old"]
    assert list((tmp_path / "atomic").glob("*.npz")) == [tmp_path / "atomic" / "snapshot.npz"]
    collection.delete("old")
    collection.save()
    assert Collection.load("atomic", storage).index.ids == ["new"]


def test_legacy_snapshot_restores_active_upsert(tmp_path):
    storage = StorageEngine(str(tmp_path))
    folder = tmp_path / "legacy"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"name": "legacy", "dim": 2}))
    np.save(folder / "vectors.npy", np.array([[1, 0], [0, 1]], dtype=np.float32))
    np.save(folder / "deleted.npy", np.array([True, False]))
    (folder / "metadata.jsonl").write_text(
        '\n'.join(json.dumps({"id": "x", "metadata": {"version": n}}) for n in [1, 2])
    )
    restored = Collection.load("legacy", storage)
    assert len(restored) == 1
    assert restored.query([0, 1])[0]["metadata"]["version"] == 2


def test_invalid_vectors_leave_existing_rows_intact():
    for index in [FlatIndex(2), HNSWIndex(2, seed=42), IVFIndex(2, nlist=2)]:
        index.add("old", [1, 0])
        for invalid in [[1, 2, 3], [float("nan"), 0], [0, 0]]:
            try:
                index.add("old", invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid vector was accepted")
        try:
            index.add_batch(["new", "new"], [[1, 0], [0, 1]])
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate batch IDs were accepted")
        assert len(index) == 1
        assert index.search([1, 0])[0]["id"] == "old"


def test_ivf_trains_on_identical_vectors():
    index = IVFIndex(2, nlist=2)
    index.add("a", [1, 0])
    index.add_batch(["b"], [[1, 0]])
    assert {row["id"] for row in index.search([1, 0], k=2)} == {"a", "b"}


def test_metrics_count_each_document_once():
    retrieved = ["a", "a", "b"]
    assert recall_at_k(retrieved, {"a", "b"}, 2) == 1.0
    assert recall_at_k(retrieved, {"a"}, 2) == 1.0
    assert average_precision(retrieved, {"a"}, 2) == 1.0
    assert ndcg_at_k(retrieved, {"a"}, 2) == 1.0
