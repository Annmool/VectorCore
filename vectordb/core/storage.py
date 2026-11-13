"""
Storage and Persistence engine for Mini Vector Database.
Includes atomic vector snapshots, legacy snapshot loading, and an operation journal.
"""

import os
import json
import tempfile
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class StorageEngine:
    """
    Manages vector data snapshots, metadata persistence, and an append-only operation journal.
    The journal is diagnostic only: it is not replayed during recovery.
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
        """Write one snapshot and publish it with a same-directory atomic replace."""
        c_dir = self._get_collection_dir(name)
        vectors = np.asarray(vectors, dtype=np.float32)
        self._validate_snapshot(config, ids, vectors, metadatas, deleted_mask)
        rows = json.dumps(list(zip(ids, metadatas)), ensure_ascii=False, allow_nan=False)
        target = os.path.join(c_dir, "snapshot.npz")
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=c_dir, suffix=".npz", delete=False) as temp:
                temp_path = temp.name
                np.savez(
                    temp, config=json.dumps(config, allow_nan=False), rows=rows,
                    vectors=vectors, deleted=np.asarray(deleted_mask, dtype=bool),
                )
                temp.flush()
                os.fsync(temp.fileno())
            os.replace(temp_path, target)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        return c_dir

    @staticmethod
    def _validate_snapshot(config, ids, vectors, metadatas, deleted_mask):
        if (vectors.shape != (len(ids), config["dim"])
                or len(metadatas) != len(ids)
                or np.asarray(deleted_mask).shape != (len(ids),)):
            raise ValueError("Snapshot row counts or dimensions do not match.")
        if not np.isfinite(vectors).all():
            raise ValueError("Snapshot contains nonfinite vectors.")
        active_ids = [doc_id for doc_id, deleted in zip(ids, deleted_mask) if not deleted]
        if len(set(active_ids)) != len(active_ids):
            raise ValueError("Snapshot contains duplicate active IDs.")

    def load_collection(
        self, name: str
    ) -> Optional[Tuple[Dict[str, Any], List[str], np.ndarray, List[Dict[str, Any]], List[bool]]]:
        """
        Load collection from disk.
        """
        c_dir = os.path.join(self.base_dir, name)
        snapshot_path = os.path.join(c_dir, "snapshot.npz")
        if os.path.exists(snapshot_path):
            with np.load(snapshot_path, allow_pickle=False) as snapshot:
                config = json.loads(snapshot["config"].item())
                rows = json.loads(snapshot["rows"].item())
                ids = [row[0] for row in rows]
                metadatas = [row[1] for row in rows]
                vectors = snapshot["vectors"]
                deleted_mask = snapshot["deleted"].tolist()
            self._validate_snapshot(config, ids, vectors, metadatas, deleted_mask)
            return config, ids, vectors, metadatas, deleted_mask

        config_path = os.path.join(c_dir, "config.json")
        vectors_path = os.path.join(c_dir, "vectors.npy")
        deleted_path = os.path.join(c_dir, "deleted.npy")
        meta_path = os.path.join(c_dir, "metadata.jsonl")

        if not any(os.path.exists(path) for path in (config_path, vectors_path, meta_path, deleted_path)):
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        vectors = np.load(vectors_path, allow_pickle=False)
        deleted_mask = np.load(deleted_path, allow_pickle=False).tolist()

        ids = []
        metadatas = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    ids.append(item["id"])
                    metadatas.append(item.get("metadata", {}))

        self._validate_snapshot(config, ids, vectors, metadatas, deleted_mask)
        return config, ids, vectors, metadatas, deleted_mask

    def append_wal(self, name: str, op: str, data: Dict[str, Any]) -> None:
        """Append an operation record for diagnostics (not a recoverable WAL)."""
        c_dir = self._get_collection_dir(name)
        wal_path = os.path.join(c_dir, "wal.log")
        entry = {"op": op, "data": data}
        with open(wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
