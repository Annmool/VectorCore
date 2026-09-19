"""
Storage and Persistence engine for Mini Vector Database.
Includes binary vector persistence, JSONL metadata store, and Write-Ahead Log (WAL).
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class StorageEngine:
    """
    Manages vector data serialization, metadata persistence, and WAL journaling.
    """

    def __init__(self, base_dir: str = "./data/collections"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_collection_dir(self, name: str) -> str:
        c_dir = os.path.join(self.base_dir, name)
        os.makedirs(c_dir, exist_ok=True)
        return c_dir

    def save_collection(
        self,
        name: str,
        ids: List[str],
        vectors: np.ndarray,
        metadatas: List[Dict[str, Any]],
        deleted_mask: List[bool],
        config: Dict[str, Any],
    ) -> str:
        """
        Save a collection to disk.
        Files saved:
        - `config.json`: collection schema, dimensions, metrics, index type
        - `vectors.npy`: raw float32/int8 vector array
        - `metadata.jsonl`: line-delimited metadata entries
        - `deleted.npy`: boolean deleted masks
        """
        c_dir = self._get_collection_dir(name)

        # 1. Config
        config_path = os.path.join(c_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # 2. Vectors
        vectors_path = os.path.join(c_dir, "vectors.npy")
        np.save(vectors_path, np.asarray(vectors, dtype=np.float32))

        # 3. Deleted mask
        deleted_path = os.path.join(c_dir, "deleted.npy")
        np.save(deleted_path, np.asarray(deleted_mask, dtype=bool))

        # 4. Metadata JSONL
        meta_path = os.path.join(c_dir, "metadata.jsonl")
        with open(meta_path, "w", encoding="utf-8") as f:
            for doc_id, meta in zip(ids, metadatas):
                line = json.dumps({"id": doc_id, "metadata": meta}, ensure_ascii=False)
                f.write(line + "\n")

        return c_dir

    def load_collection(
        self, name: str
    ) -> Optional[Tuple[Dict[str, Any], List[str], np.ndarray, List[Dict[str, Any]], List[bool]]]:
        """
        Load collection from disk.
        """
        c_dir = os.path.join(self.base_dir, name)
        config_path = os.path.join(c_dir, "config.json")
        vectors_path = os.path.join(c_dir, "vectors.npy")
        deleted_path = os.path.join(c_dir, "deleted.npy")
        meta_path = os.path.join(c_dir, "metadata.jsonl")

        if not os.path.exists(config_path) or not os.path.exists(vectors_path):
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        vectors = np.load(vectors_path)
        deleted_mask = np.load(deleted_path).tolist() if os.path.exists(deleted_path) else [False] * len(vectors)

        ids = []
        metadatas = []
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        ids.append(item["id"])
                        metadatas.append(item.get("metadata", {}))

        return config, ids, vectors, metadatas, deleted_mask

    def append_wal(self, name: str, op: str, data: Dict[str, Any]) -> None:
        """Write operation to Write-Ahead Log (WAL) for durability."""
        c_dir = self._get_collection_dir(name)
        wal_path = os.path.join(c_dir, "wal.log")
        entry = {"op": op, "data": data}
        with open(wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
