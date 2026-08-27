"""
GPU-accelerated face embedding search via FAISS (GpuIndexFlatL2).

Falls back to NumPy linear scan when CUDA/FAISS-GPU is unavailable.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("gpu_face_index")

_index_lock = threading.Lock()
_gpu_index = None
_cpu_index = None
_id_map: List[int] = []
_dim: int = 0
_gpu_ready: bool = False


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return (matrix / norms).astype(np.float32)


def gpu_index_available() -> bool:
    try:
        import faiss

        return bool(hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0)
    except Exception:
        return False


def _flatten_embedding(feat) -> Optional[np.ndarray]:
    try:
        flat = np.asarray(feat, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        return None
    return flat


def _select_embeddings(
    raw_entries: List[Tuple[np.ndarray, int]],
) -> Tuple[List[np.ndarray], List[int], int]:
    """
    Keep embeddings that share the dominant vector length.
    Mixed OpenCV/ArcFace dimensions in SQLite cannot be stacked for FAISS.
    """
    if not raw_entries:
        return [], [], 0

    dim_counts = Counter(flat.shape[0] for flat, _ in raw_entries)
    target_dim = dim_counts.most_common(1)[0][0]

    vectors: List[np.ndarray] = []
    ids: List[int] = []
    skipped = 0
    for flat, person_id in raw_entries:
        if flat.shape[0] != target_dim:
            skipped += 1
            continue
        vectors.append(flat)
        ids.append(person_id)
    return vectors, ids, skipped


def rebuild_face_index(
    conn,
    *,
    force: bool = False,
) -> int:
    """Rebuild FAISS index from SQLite gallery embeddings. Returns vector count."""
    global _gpu_index, _cpu_index, _id_map, _dim, _gpu_ready

    from identity import _collect_person_face_embeddings

    with _index_lock:
        try:
            by_person = _collect_person_face_embeddings(conn)
            raw_entries: List[Tuple[np.ndarray, int]] = []
            for person_id, entries in sorted(by_person.items()):
                for feat, _quality, _source in entries:
                    flat = _flatten_embedding(feat)
                    if flat is None:
                        continue
                    raw_entries.append((flat, int(person_id)))

            vectors, ids, skipped = _select_embeddings(raw_entries)
            if skipped:
                log.warning(
                    "FAISS index skipped %d embedding(s) with non-uniform dimensions",
                    skipped,
                )

            if not vectors:
                _gpu_index = None
                _cpu_index = None
                _id_map = []
                _dim = 0
                _gpu_ready = False
                return 0

            matrix = _normalize_rows(np.stack(vectors, axis=0))
            _dim = int(matrix.shape[1])
            _id_map = ids

            import faiss

            cpu_index = faiss.IndexFlatL2(_dim)
            cpu_index.add(matrix)
            _cpu_index = cpu_index

            if gpu_index_available():
                res = faiss.StandardGpuResources()
                gpu_index = faiss.GpuIndexFlatL2(res, _dim)
                gpu_index.add(matrix)
                _gpu_index = gpu_index
                _gpu_ready = True
                log.info(
                    "FAISS GPU index ready — %d embeddings, dim=%d",
                    len(_id_map),
                    _dim,
                )
            else:
                _gpu_index = None
                _gpu_ready = False
                log.info(
                    "FAISS CPU index ready — %d embeddings, dim=%d",
                    len(_id_map),
                    _dim,
                )
        except Exception as exc:
            log.warning("FAISS index build failed (%s) — using linear fallback", exc)
            _gpu_index = None
            _cpu_index = None
            _id_map = []
            _dim = 0
            _gpu_ready = False
            return 0

        return len(_id_map)


def search_best(
    query: np.ndarray,
    *,
    top_k: int = 8,
) -> List[Tuple[int, float]]:
    """
    Return [(person_id, l2_distance)] sorted best-first.
    Query should be a normalized face embedding vector.
    """
    flat = np.asarray(query, dtype=np.float32).reshape(1, -1)
    if flat.size == 0:
        return []
    flat = _normalize_rows(flat)

    with _index_lock:
        if not _id_map or _dim == 0 or flat.shape[1] != _dim:
            return []

        k = min(top_k, len(_id_map))
        try:
            index = _gpu_index if _gpu_ready and _gpu_index is not None else _cpu_index
            if index is None:
                return []
            distances, indices = index.search(flat, k)
            out: List[Tuple[int, float]] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(_id_map):
                    continue
                out.append((_id_map[int(idx)], float(dist)))
            return out
        except Exception as exc:
            log.debug("FAISS search failed: %s", exc)
            return []
