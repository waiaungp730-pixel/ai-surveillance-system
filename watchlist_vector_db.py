"""
Watchlist face embeddings via Qdrant (+ in-process memory mirror).

Designed for ArcFace / InsightFace 512-d vectors (also accepts other dims).
Default storage is a local on-disk Qdrant path — no Docker required.
Optional remote: set QDRANT_URL=http://localhost:6333

Public API
----------
  upsert_face_embedding(...)     insert/update one vector + metadata
  search_similar_faces(...)      cosine top-k above threshold
  sync_watchlist_from_sqlite(...)  pull gallery → Qdrant + memory index
  start_watchlist_sync_daemon(...) hot-reload without restarting trackers
  notify_watchlist_changed()     signal other processes to resync
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("watchlist_vector_db")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_QDRANT_PATH = PROJECT_DIR / "data" / "qdrant_watchlist"
DEFAULT_COLLECTION = "watchlist_faces"
DEFAULT_VECTOR_SIZE = 512
DEFAULT_SCORE_THRESHOLD = 0.50
DEFAULT_TOP_K = 5
SYNC_STAMP_PATH = PROJECT_DIR / "live_preview" / "watchlist_vector.sync"
SYNC_POLL_SEC = 2.0

# Optional overrides via environment
ENV_QDRANT_URL = "QDRANT_URL"
ENV_QDRANT_PATH = "QDRANT_PATH"
ENV_QDRANT_COLLECTION = "QDRANT_COLLECTION"
ENV_SCORE_THRESHOLD = "WATCHLIST_SCORE_THRESHOLD"


@dataclass(frozen=True)
class FaceMatch:
    """One cosine-similarity hit from the watchlist index."""

    person_id: int
    person_name: str
    role: str
    score: float
    point_id: str
    added_date: str = ""
    source: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FaceVectorRecord:
    """Payload stored alongside each embedding in Qdrant."""

    person_id: int
    person_name: str
    role: str = "watchlist"
    added_date: str = ""
    source: str = "manual"
    encoding_slot: str = "primary"
    quality_score: Optional[float] = None

    def payload(self) -> Dict[str, Any]:
        return {
            "person_id": int(self.person_id),
            "person_name": str(self.person_name or ""),
            "role": str(self.role or "watchlist"),
            "added_date": str(self.added_date or _utc_now_iso()),
            "source": str(self.source or ""),
            "encoding_slot": str(self.encoding_slot or "primary"),
            "quality_score": (
                float(self.quality_score) if self.quality_score is not None else None
            ),
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_vector(vec: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Face embedding is empty or non-finite")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        raise ValueError("Face embedding has near-zero L2 norm")
    return (arr / norm).astype(np.float32)


def _point_id(person_id: int, encoding_slot: str) -> str:
    """Deterministic UUID so upserts replace the same gallery slot."""
    key = f"watchlist-face:{int(person_id)}:{encoding_slot}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def default_score_threshold() -> float:
    raw = (
        os.environ.get(ENV_SCORE_THRESHOLD)
        or os.environ.get("TELEGRAM_ALERT_MIN_PERCENT")
        or os.environ.get("WATCHLIST_ALERT_MIN_PERCENT")
        or ""
    ).strip()
    if raw:
        try:
            value = float(raw)
            if value > 1.0:
                value = value / 100.0
            return max(0.0, min(1.0, value))
        except ValueError:
            pass
    return DEFAULT_SCORE_THRESHOLD


# ── In-process memory index (hot path during tracking) ───────────────────────


class MemoryFaceIndex:
    """
    Thread-safe L2-normalized matrix kept in RAM for microsecond cosine search.

    Synced from Qdrant / SQLite by WatchlistVectorSync so trackers never need a
    process restart when the watchlist grows.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._matrix: Optional[np.ndarray] = None  # (N, D)
        self._meta: List[Dict[str, Any]] = []
        self._dim: int = 0
        self._version: float = 0.0

    @property
    def size(self) -> int:
        with self._lock:
            return 0 if self._matrix is None else int(self._matrix.shape[0])

    @property
    def dim(self) -> int:
        with self._lock:
            return self._dim

    @property
    def version(self) -> float:
        with self._lock:
            return self._version

    def clear(self) -> None:
        with self._lock:
            self._matrix = None
            self._meta = []
            self._dim = 0
            self._version = time.time()

    def rebuild(
        self,
        vectors: Sequence[np.ndarray],
        metadata: Sequence[Dict[str, Any]],
    ) -> int:
        if len(vectors) != len(metadata):
            raise ValueError("vectors/metadata length mismatch")
        with self._lock:
            if not vectors:
                self.clear()
                return 0
            mats = [_normalize_vector(v) for v in vectors]
            # Prefer the dominant embedding size (InsightFace=512). Skip legacy
            # OpenCV/SFace 128-d rows instead of failing the whole watchlist index.
            from collections import Counter

            dim_counts = Counter(int(m.shape[0]) for m in mats)
            dim = int(dim_counts.most_common(1)[0][0])
            kept_mats: List[np.ndarray] = []
            kept_meta: List[Dict[str, Any]] = []
            skipped = 0
            for m, row in zip(mats, metadata):
                if int(m.shape[0]) != dim:
                    skipped += 1
                    continue
                kept_mats.append(m)
                kept_meta.append(dict(row))
            if skipped:
                log.warning(
                    "Watchlist memory index skipped %d embedding(s) with non-uniform dims (kept dim=%d)",
                    skipped,
                    dim,
                )
            if not kept_mats:
                self.clear()
                return 0
            self._matrix = np.stack(kept_mats, axis=0)
            self._meta = kept_meta
            self._dim = dim
            self._version = time.time()
            return int(self._matrix.shape[0])

    def upsert_row(self, vector: np.ndarray, meta: Dict[str, Any]) -> None:
        """Insert or replace one row keyed by point_id (small galleries)."""
        vec = _normalize_vector(vector)
        point_id = str(meta.get("point_id") or "")
        with self._lock:
            if self._matrix is None or self._matrix.size == 0:
                self._matrix = vec.reshape(1, -1)
                self._meta = [dict(meta)]
                self._dim = int(vec.shape[0])
                self._version = time.time()
                return
            if int(vec.shape[0]) != self._dim:
                raise ValueError(
                    f"Embedding dim {vec.shape[0]} != index dim {self._dim}"
                )
            replace_at = -1
            if point_id:
                for i, row in enumerate(self._meta):
                    if str(row.get("point_id") or "") == point_id:
                        replace_at = i
                        break
            if replace_at >= 0:
                self._matrix[replace_at] = vec
                self._meta[replace_at] = dict(meta)
            else:
                self._matrix = np.vstack([self._matrix, vec.reshape(1, -1)])
                self._meta.append(dict(meta))
            self._version = time.time()

    def search(
        self,
        query: Union[np.ndarray, Sequence[float]],
        *,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> List[FaceMatch]:
        q = _normalize_vector(query)
        with self._lock:
            if self._matrix is None or not self._meta:
                return []
            if int(q.shape[0]) != self._dim:
                return []
            # Cosine similarity for L2-normalized vectors == dot product.
            scores = self._matrix @ q
            k = max(1, min(int(top_k), scores.shape[0]))
            # argpartition then sort the top slice
            if k < scores.shape[0]:
                idx_part = np.argpartition(-scores, k - 1)[:k]
            else:
                idx_part = np.arange(scores.shape[0])
            idx_sorted = idx_part[np.argsort(-scores[idx_part])]

            out: List[FaceMatch] = []
            for i in idx_sorted:
                score = float(scores[int(i)])
                if score < float(score_threshold):
                    continue
                row = self._meta[int(i)]
                out.append(
                    FaceMatch(
                        person_id=int(row.get("person_id") or 0),
                        person_name=str(row.get("person_name") or ""),
                        role=str(row.get("role") or "watchlist"),
                        score=score,
                        point_id=str(row.get("point_id") or ""),
                        added_date=str(row.get("added_date") or ""),
                        source=str(row.get("source") or ""),
                        extra={
                            k: v
                            for k, v in row.items()
                            if k
                            not in {
                                "person_id",
                                "person_name",
                                "role",
                                "added_date",
                                "source",
                                "point_id",
                            }
                        },
                    )
                )
            return out


_MEMORY_INDEX = MemoryFaceIndex()


def get_memory_index() -> MemoryFaceIndex:
    return _MEMORY_INDEX


# ── Qdrant store ─────────────────────────────────────────────────────────────


class QdrantFaceStore:
    """Persistent cosine index for watchlist face embeddings."""

    def __init__(
        self,
        *,
        collection: Optional[str] = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        path: Optional[Union[str, Path]] = None,
        url: Optional[str] = None,
    ) -> None:
        self.collection = (
            collection
            or os.environ.get(ENV_QDRANT_COLLECTION, "").strip()
            or DEFAULT_COLLECTION
        )
        self.vector_size = int(vector_size)
        self._url = (url or os.environ.get(ENV_QDRANT_URL, "")).strip() or None
        env_path = os.environ.get(ENV_QDRANT_PATH, "").strip()
        self._path = Path(path or env_path or DEFAULT_QDRANT_PATH)
        self._client = None
        self._lock = threading.RLock()
        self._ensured_dim: Optional[int] = None

    def _import_client(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as rest_models
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for watchlist vector search.\n"
                "  Install: py -3.12 -m pip install qdrant-client"
            ) from exc
        return QdrantClient, rest_models

    def connect(self):
        with self._lock:
            if self._client is not None:
                return self._client
            QdrantClient, _ = self._import_client()
            if self._url:
                self._client = QdrantClient(url=self._url, prefer_grpc=False)
                log.info("Qdrant connected (remote): %s", self._url)
            else:
                self._path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(self._path))
                log.info("Qdrant connected (local): %s", self._path)
            return self._client

    def ensure_collection(self, vector_size: Optional[int] = None) -> None:
        _, rest_models = self._import_client()
        client = self.connect()
        dim = int(vector_size or self.vector_size)
        with self._lock:
            names = {c.name for c in client.get_collections().collections}
            if self.collection not in names:
                client.create_collection(
                    collection_name=self.collection,
                    vectors_config=rest_models.VectorParams(
                        size=dim,
                        distance=rest_models.Distance.COSINE,
                    ),
                )
                log.info(
                    "Created Qdrant collection '%s' (dim=%d, cosine)",
                    self.collection,
                    dim,
                )
            self._ensured_dim = dim
            self.vector_size = dim

    def upsert(
        self,
        embedding: Union[np.ndarray, Sequence[float]],
        record: FaceVectorRecord,
        *,
        point_id: Optional[str] = None,
    ) -> str:
        """Insert or replace one face embedding. Returns point id."""
        vec = _normalize_vector(embedding)
        self.ensure_collection(vector_size=int(vec.shape[0]))
        if int(vec.shape[0]) != self.vector_size:
            raise ValueError(
                f"Embedding dim {vec.shape[0]} != collection dim {self.vector_size}"
            )

        _, rest_models = self._import_client()
        client = self.connect()
        pid = point_id or _point_id(record.person_id, record.encoding_slot)
        payload = record.payload()
        payload["point_id"] = pid

        with self._lock:
            client.upsert(
                collection_name=self.collection,
                points=[
                    rest_models.PointStruct(
                        id=pid,
                        vector=vec.tolist(),
                        payload=payload,
                    )
                ],
                wait=True,
            )
        # Keep RAM mirror warm for live tracking.
        _MEMORY_INDEX.upsert_row(vec, payload)
        return pid

    def upsert_many(
        self,
        items: Sequence[Tuple[np.ndarray, FaceVectorRecord]],
    ) -> int:
        if not items:
            _MEMORY_INDEX.clear()
            return 0
        first = _normalize_vector(items[0][0])
        # Always warm the in-process index first — multi-camera local Qdrant
        # cannot share one storage folder, so memory is the live match path.
        mem_vecs: List[np.ndarray] = []
        mem_meta: List[Dict[str, Any]] = []
        points = []
        _, rest_models = self._import_client()
        for emb, record in items:
            vec = _normalize_vector(emb)
            if int(vec.shape[0]) != int(first.shape[0]):
                log.warning(
                    "Skipping person_id=%s — dim %d != %d",
                    record.person_id,
                    vec.shape[0],
                    first.shape[0],
                )
                continue
            pid = _point_id(record.person_id, record.encoding_slot)
            payload = record.payload()
            payload["point_id"] = pid
            points.append(
                rest_models.PointStruct(id=pid, vector=vec.tolist(), payload=payload)
            )
            mem_vecs.append(vec)
            mem_meta.append(payload)

        count = _MEMORY_INDEX.rebuild(mem_vecs, mem_meta) if mem_vecs else 0
        if not points:
            return 0

        try:
            self.ensure_collection(vector_size=int(first.shape[0]))
            client = self.connect()
            with self._lock:
                chunk = 256
                for i in range(0, len(points), chunk):
                    client.upsert(
                        collection_name=self.collection,
                        points=points[i : i + chunk],
                        wait=True,
                    )
        except Exception as exc:
            log.warning(
                "Qdrant upsert skipped (memory index has %d vector(s)): %s",
                count,
                exc,
            )
        return count

    def search(
        self,
        query: Union[np.ndarray, Sequence[float]],
        *,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        role: Optional[str] = None,
    ) -> List[FaceMatch]:
        """Cosine similarity search; returns matches with score >= threshold."""
        vec = _normalize_vector(query)
        self.ensure_collection(vector_size=int(vec.shape[0]))
        if int(vec.shape[0]) != self.vector_size:
            return []

        _, rest_models = self._import_client()
        client = self.connect()
        query_filter = None
        if role:
            query_filter = rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="role",
                        match=rest_models.MatchValue(value=str(role)),
                    )
                ]
            )

        with self._lock:
            hits = client.search(
                collection_name=self.collection,
                query_vector=vec.tolist(),
                limit=max(1, int(top_k)),
                score_threshold=float(score_threshold),
                query_filter=query_filter,
                with_payload=True,
            )

        out: List[FaceMatch] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            out.append(
                FaceMatch(
                    person_id=int(payload.get("person_id") or 0),
                    person_name=str(payload.get("person_name") or ""),
                    role=str(payload.get("role") or "watchlist"),
                    score=float(hit.score),
                    point_id=str(hit.id),
                    added_date=str(payload.get("added_date") or ""),
                    source=str(payload.get("source") or ""),
                    extra=payload,
                )
            )
        return out

    def delete_person(self, person_id: int) -> int:
        """Remove all points for a person_id. Returns deleted count (best-effort)."""
        _, rest_models = self._import_client()
        client = self.connect()
        with self._lock:
            client.delete(
                collection_name=self.collection,
                points_selector=rest_models.FilterSelector(
                    filter=rest_models.Filter(
                        must=[
                            rest_models.FieldCondition(
                                key="person_id",
                                match=rest_models.MatchValue(value=int(person_id)),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        return 1

    def count(self) -> int:
        client = self.connect()
        with self._lock:
            info = client.get_collection(self.collection)
        return int(getattr(info, "points_count", 0) or 0)


_STORE: Optional[QdrantFaceStore] = None
_STORE_LOCK = threading.Lock()


def get_face_store(**kwargs) -> QdrantFaceStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = QdrantFaceStore(**kwargs)
        return _STORE


# ── Public helpers ───────────────────────────────────────────────────────────


def upsert_face_embedding(
    embedding: Union[np.ndarray, Sequence[float]],
    *,
    person_name: str,
    person_id: int,
    role: str = "watchlist",
    added_date: Optional[str] = None,
    source: str = "manual",
    encoding_slot: str = "primary",
    quality_score: Optional[float] = None,
    store: Optional[QdrantFaceStore] = None,
) -> str:
    """
    Insert or update one face embedding (typically 512-d ArcFace) with metadata.

    Returns the Qdrant point id (stable across upserts for the same slot).
    """
    db = store or get_face_store()
    record = FaceVectorRecord(
        person_id=int(person_id),
        person_name=str(person_name or f"Person_{person_id}"),
        role=str(role or "watchlist"),
        added_date=added_date or _utc_now_iso(),
        source=source,
        encoding_slot=encoding_slot,
        quality_score=quality_score,
    )
    return db.upsert(embedding, record)


def search_similar_faces(
    query_embedding: Union[np.ndarray, Sequence[float]],
    *,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: Optional[float] = None,
    prefer_memory: bool = True,
    role: Optional[str] = None,
    store: Optional[QdrantFaceStore] = None,
) -> List[FaceMatch]:
    """
    Fast cosine similarity search against the watchlist.

    Returns top matches with score >= ``score_threshold`` (default 0.65).
    Prefers the in-process memory index when warm; falls back to Qdrant.
    """
    threshold = (
        float(score_threshold)
        if score_threshold is not None
        else default_score_threshold()
    )

    if prefer_memory and _MEMORY_INDEX.size > 0:
        hits = _MEMORY_INDEX.search(
            query_embedding, top_k=top_k, score_threshold=threshold
        )
        if role:
            hits = [h for h in hits if h.role == role]
        if hits or _MEMORY_INDEX.size > 0:
            return hits

    db = store or get_face_store()
    return db.search(
        query_embedding,
        top_k=top_k,
        score_threshold=threshold,
        role=role,
    )


# ── SQLite gallery sync ──────────────────────────────────────────────────────


def _load_gallery_rows(
    conn: sqlite3.Connection,
) -> List[Tuple[np.ndarray, FaceVectorRecord]]:
    """Flatten known_persons + encodings into upsert payloads."""
    from identity import (
        _collect_person_face_embeddings,
        ensure_known_person_name_column,
        resolve_known_person_name,
    )

    ensure_known_person_name_column(conn)
    by_person = _collect_person_face_embeddings(conn)
    items: List[Tuple[np.ndarray, FaceVectorRecord]] = []

    # Optional role column (ignore if absent).
    cols = {row[1] for row in conn.execute("PRAGMA table_info(known_persons)")}
    has_role = "role" in cols
    role_map: Dict[int, str] = {}
    created_map: Dict[int, str] = {}
    for row in conn.execute(
        "SELECT id, created_at"
        + (", role" if has_role else "")
        + " FROM known_persons"
    ):
        pid = int(row[0])
        created_map[pid] = str(row[1] or "")
        if has_role:
            role_map[pid] = str(row[2] or "watchlist")

    for person_id, entries in sorted(by_person.items()):
        name = resolve_known_person_name(conn, person_id) or f"Person_{person_id}"
        role = role_map.get(int(person_id), "watchlist")
        added = created_map.get(int(person_id)) or _utc_now_iso()
        for idx, (feat, quality, source) in enumerate(entries):
            try:
                vec = _normalize_vector(feat)
            except ValueError:
                continue
            slot = f"{source}-{idx}"
            items.append(
                (
                    vec,
                    FaceVectorRecord(
                        person_id=int(person_id),
                        person_name=str(name),
                        role=role,
                        added_date=added,
                        source=str(source),
                        encoding_slot=slot,
                        quality_score=(
                            float(quality) if quality is not None else None
                        ),
                    ),
                )
            )
    return items


def notify_watchlist_changed() -> None:
    """
    Touch the cross-process sync stamp so running trackers reload the index
    without a restart of ``run_system.py``.
    """
    try:
        SYNC_STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        SYNC_STAMP_PATH.write_text(
            f"{time.time():.6f}\n{datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        log.debug("Failed to write watchlist sync stamp: %s", exc)


def _rebuild_memory_from_items(
    items: Sequence[Tuple[np.ndarray, FaceVectorRecord]],
) -> int:
    """Build the in-process cosine index from SQLite gallery rows (no Qdrant)."""
    if not items:
        _MEMORY_INDEX.clear()
        return 0
    mem_vecs: List[np.ndarray] = []
    mem_meta: List[Dict[str, Any]] = []
    for emb, record in items:
        try:
            vec = _normalize_vector(emb)
        except ValueError:
            continue
        pid = _point_id(record.person_id, record.encoding_slot)
        payload = record.payload()
        payload["point_id"] = pid
        mem_vecs.append(vec)
        mem_meta.append(payload)
    if not mem_vecs:
        _MEMORY_INDEX.clear()
        return 0
    return _MEMORY_INDEX.rebuild(mem_vecs, mem_meta)


def sync_watchlist_from_sqlite(
    conn: Optional[sqlite3.Connection] = None,
    *,
    db_path: Optional[Union[str, Path]] = None,
    store: Optional[QdrantFaceStore] = None,
    notify: bool = True,
) -> int:
    """
    Rebuild in-memory watchlist index from SQLite (Qdrant optional).

    Local Qdrant cannot be opened by multiple camera processes — memory index
    is the reliable live path. Returns number of embeddings loaded into memory.
    """
    owns_conn = False
    if conn is None:
        path = Path(db_path) if db_path else PROJECT_DIR / "visitor_tracking.db"
        if not path.is_file():
            log.warning("Watchlist sync skipped — DB missing: %s", path)
            return 0
        conn = sqlite3.connect(str(path), timeout=30.0)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass
        owns_conn = True

    try:
        items = _load_gallery_rows(conn)
        count = _rebuild_memory_from_items(items)
        # Best-effort durable index — never fail the live memory sync on lock.
        if items:
            try:
                db = store or get_face_store()
                db.upsert_many(items)
            except Exception as exc:
                log.warning(
                    "Qdrant watchlist sync skipped (memory has %d): %s",
                    count,
                    exc,
                )
        # Keep FAISS GPU gallery in this process fresh for fallback search.
        try:
            from gpu_face_index import rebuild_face_index

            rebuild_face_index(conn)
        except Exception as exc:
            log.debug("FAISS rebuild after watchlist sync skipped: %s", exc)
        log.info(
            "Watchlist vector sync complete — %d embedding(s) in memory",
            count,
        )
        if notify:
            notify_watchlist_changed()
        return count
    finally:
        if owns_conn and conn is not None:
            conn.close()


# ── Hot-reload daemon (no tracker restart) ───────────────────────────────────


class WatchlistVectorSync:
    """
    Background thread: watches the sync stamp (and optional DB mtime) and
    refreshes Qdrant + the in-memory cosine index while trackers keep running.
    """

    def __init__(
        self,
        *,
        db_path: Optional[Union[str, Path]] = None,
        poll_sec: float = SYNC_POLL_SEC,
        store: Optional[QdrantFaceStore] = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else PROJECT_DIR / "visitor_tracking.db"
        self.poll_sec = max(0.5, float(poll_sec))
        self.store = store
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_stamp_mtime: float = 0.0
        self._last_db_mtime: float = 0.0

    def start(self) -> "WatchlistVectorSync":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        # Initial load
        try:
            sync_watchlist_from_sqlite(
                db_path=self.db_path, store=self.store, notify=False
            )
        except Exception as exc:
            log.warning("Initial watchlist vector sync failed: %s", exc)
        self._last_stamp_mtime = _safe_mtime(SYNC_STAMP_PATH)
        self._last_db_mtime = _safe_mtime(self.db_path)
        self._thread = threading.Thread(
            target=self._run,
            name="watchlist-vector-sync",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "Watchlist vector sync daemon started (poll=%.1fs, db=%s)",
            self.poll_sec,
            self.db_path.name,
        )
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_sec):
            try:
                stamp_m = _safe_mtime(SYNC_STAMP_PATH)
                db_m = _safe_mtime(self.db_path)
                changed = False
                if stamp_m > self._last_stamp_mtime + 1e-6:
                    self._last_stamp_mtime = stamp_m
                    changed = True
                # Also catch direct SQLite edits that forgot to notify.
                if db_m > self._last_db_mtime + 1e-6:
                    self._last_db_mtime = db_m
                    changed = True
                if not changed:
                    continue
                n = sync_watchlist_from_sqlite(
                    db_path=self.db_path, store=self.store, notify=False
                )
                log.info("Hot-reloaded watchlist vectors (%d) — no restart needed", n)
            except Exception as exc:
                log.warning("Watchlist hot-reload failed: %s", exc)


_SYNC_DAEMON: Optional[WatchlistVectorSync] = None
_SYNC_DAEMON_LOCK = threading.Lock()


def start_watchlist_sync_daemon(
    *,
    db_path: Optional[Union[str, Path]] = None,
    poll_sec: float = SYNC_POLL_SEC,
) -> WatchlistVectorSync:
    """Idempotent start of the cross-process hot-reload daemon."""
    global _SYNC_DAEMON
    with _SYNC_DAEMON_LOCK:
        if _SYNC_DAEMON is None:
            _SYNC_DAEMON = WatchlistVectorSync(db_path=db_path, poll_sec=poll_sec)
            _SYNC_DAEMON.start()
        return _SYNC_DAEMON


def stop_watchlist_sync_daemon() -> None:
    global _SYNC_DAEMON
    with _SYNC_DAEMON_LOCK:
        if _SYNC_DAEMON is not None:
            _SYNC_DAEMON.stop()
            _SYNC_DAEMON = None


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime) if path.is_file() else 0.0
    except OSError:
        return 0.0


# ── Distance bridge for existing FAISS / cosine-distance callers ─────────────


def cosine_similarity_to_distance(score: float) -> float:
    """Map cosine similarity [−1,1] → distance-like [0,2] (0 = identical)."""
    return float(max(0.0, 1.0 - float(score)))


def search_best_person_ids(
    query_embedding: Union[np.ndarray, Sequence[float]],
    *,
    top_k: int = 8,
    score_threshold: Optional[float] = None,
) -> List[Tuple[int, float]]:
    """
    Compatibility helper similar to ``gpu_face_index.search_best``.

    Returns ``[(person_id, cosine_distance)]`` sorted best-first
    (distance = 1 − cosine_similarity).
    """
    hits = search_similar_faces(
        query_embedding,
        top_k=top_k,
        score_threshold=score_threshold
        if score_threshold is not None
        else default_score_threshold(),
    )
    return [(h.person_id, cosine_similarity_to_distance(h.score)) for h in hits]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Smoke test with a random unit vector (no real faces).
    rng = np.random.default_rng(0)
    demo = rng.normal(size=512).astype(np.float32)
    demo /= np.linalg.norm(demo) + 1e-8
    pid = upsert_face_embedding(
        demo,
        person_name="Demo Watchlist",
        person_id=900001,
        role="watchlist",
    )
    print("upserted", pid)
    matches = search_similar_faces(demo, top_k=3, score_threshold=0.5)
    for m in matches:
        print(f"  match id={m.person_id} name={m.person_name} score={m.score:.4f}")
