"""
Parser Monstro - Serviço de extração de texto de documentos de licitação.
API REST FastAPI — roda em container isolado na porta 7000.
"""

import asyncio
import base64
import json
import logging
import subprocess
from collections import deque
import os
import re
import requests
import sqlite3
import threading
import time
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

import httpx
import magic
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel

from zip_recursive import ZipExtractionLimits, extract_zip_recursive, write_extracted_files

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
PARSER_MODE = os.getenv("PARSER_MODE", "balanced").strip().lower()
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1" if PARSER_MODE == "precision_first" else "2"))
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "/app/storage"))
JOBS_DB_PATH = STORAGE_ROOT / "jobs.sqlite3"

# Fallback chain thresholds:
# 1. pymupdf           → score >= 0.90, pronto
# 2. paddleocr-vl-1.5  → score >= 0.82, pronto
# 3. tesseract         → score >= 0.50, pronto
# 4. abaixo disso      → registra falha + salva páginas para revisão manual
PYMUPDF_QUALITY_THRESHOLD = float(os.getenv("PYMUPDF_QUALITY_THRESHOLD", "0.9"))
PADDLE_OCR_QUALITY_THRESHOLD = float(os.getenv("PADDLE_OCR_QUALITY_THRESHOLD", "0.82"))
TESSERACT_QUALITY_THRESHOLD = float(os.getenv("TESSERACT_QUALITY_THRESHOLD", "0.5"))
MIN_CHARS_PER_PAGE_NATIVE = int(os.getenv("MIN_CHARS_PER_PAGE_NATIVE", "80"))
CLEAN_OCR_NOISE = os.getenv("CLEAN_OCR_NOISE", "true").lower() == "true"
PADDLE_OCR_URL = os.getenv("PADDLE_OCR_URL", "http://127.0.0.1:8100")
PADDLE_OCR_MODEL = os.getenv("PADDLE_OCR_MODEL", "PaddlePaddle/PaddleOCR-VL-1.5")
PADDLE_OCR_TIMEOUT = int(os.getenv("PADDLE_OCR_TIMEOUT", "120"))
PADDLE_OCR_GPU_UTIL = float(os.getenv("PADDLE_OCR_GPU_UTIL", "0.85"))
PADDLE_OCR_IMAGE_DPI = int(os.getenv("PADDLE_OCR_IMAGE_DPI", "200"))
REVIEW_ROOT = STORAGE_ROOT / "review"

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("lobeira")

SYSTEM_LOG_HISTORY_LIMIT = int(os.getenv("SYSTEM_LOG_HISTORY_LIMIT", "500"))
SYSTEM_LOG_FILE_PATH = os.getenv("PARSER_SYSTEM_LOG_PATH", "").strip()
SYSTEMD_UNIT = os.getenv("PARSER_SYSTEMD_UNIT", os.getenv("SYSTEMD_UNIT", "")).strip()
SYSTEMD_INVOCATION_ID = os.getenv("INVOCATION_ID", "").strip()
SYSTEM_LOG_ACCESS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r'\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+/(health|queue|jobs|logs)(?:\b|/)',
        r'\b/health\b',
        r'\b(liveness|readiness|healthcheck)\b',
        r'\b127\.0\.0\.1:\d+ - "(?:GET|HEAD) /',
    )
]
_system_log_buffer: Deque[Dict[str, Any]] = deque(maxlen=max(50, SYSTEM_LOG_HISTORY_LIMIT))
_system_log_buffer_lock = threading.Lock()


class _SystemLogBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            created_at = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
            entry = {
                "timestamp": created_at,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "source": "in_memory",
            }
            with _system_log_buffer_lock:
                _system_log_buffer.append(entry)
        except Exception:
            pass


def _ensure_system_log_capture_handler() -> None:
    for handler in logger.handlers:
        if isinstance(handler, _SystemLogBufferHandler):
            return
    logger.addHandler(_SystemLogBufferHandler())


_ensure_system_log_capture_handler()


class PaddleOCRVLParser:
    def __init__(self, base_url: str, model: str, timeout_s: int = 120, image_dpi: int = 200) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.image_dpi = image_dpi

    def parse_pdf(self, path: Path, review_dir: Optional[Path] = None) -> Dict[str, Any]:
        import fitz
        from openai import OpenAI

        review_dir = review_dir or REVIEW_ROOT / f"adhoc-{uuid.uuid4().hex}"
        review_dir.mkdir(parents=True, exist_ok=True)

        client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=os.getenv("PADDLE_OCR_API_KEY", "dummy"),
            timeout=self.timeout_s,
        )
        doc = fitz.open(str(path))
        parts: List[str] = []
        logs: List[str] = []
        failed_pages: List[int] = []
        total_chars = 0
        pages = doc.page_count

        try:
            for page_index in range(pages):
                page_number = page_index + 1
                page = doc.load_page(page_index)
                pix = page.get_pixmap(dpi=self.image_dpi, alpha=False)
                image_bytes = pix.tobytes("png")
                image_b64 = base64.b64encode(image_bytes).decode("ascii")
                page_review_path = review_dir / f"page-{page_number:04d}.png"

                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        temperature=0,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            "Extraia fielmente todo o texto visível desta página em português. "
                                            "Mantenha a ordem de leitura, preserve números e pontuação, "
                                            "não resuma e não invente conteúdo."
                                        ),
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                                    },
                                ],
                            }
                        ],
                    )
                    content = response.choices[0].message.content if response.choices else ""
                    page_text = _normalize_text(content or "")
                    char_count = len(page_text)
                    total_chars += char_count
                    if char_count == 0:
                        failed_pages.append(page_number)
                        page_review_path.write_bytes(image_bytes)
                        logs.append(f"paddleocr_vl: página {page_number} vazia")
                    else:
                        parts.append(page_text)
                        logs.append(f"paddleocr_vl: página {page_number} ok ({char_count} chars)")
                except Exception as exc:
                    failed_pages.append(page_number)
                    page_review_path.write_bytes(image_bytes)
                    message = str(exc)[:300]
                    if "out of memory" in message.lower() or "oom" in message.lower():
                        message = f"OOM: {message}"
                    elif "timed out" in message.lower() or "timeout" in message.lower():
                        message = f"timeout: {message}"
                    else:
                        message = f"erro: {message}"
                    logs.append(f"paddleocr_vl: página {page_number} falhou ({message})")
        finally:
            doc.close()

        text = "\n\n".join(parts).strip()
        quality = _quality_score(text, pages) if text else 0.0
        avg_chars_per_page = round(total_chars / max(1, pages), 2)
        logs.append(f"paddleocr_vl: total={total_chars} chars, {pages}p, avg_chars_per_page={avg_chars_per_page}, score={quality:.2f}")
        return {
            "text": text,
            "pages": pages,
            "quality": quality,
            "logs": logs,
            "failed_pages": failed_pages,
            "avg_chars_per_page": avg_chars_per_page,
            "saved_review_dir": str(review_dir),
        }


# ---------------------------------------------------------------------------
# Durable job store (SQLite)
# ---------------------------------------------------------------------------
jobs: Dict[str, Dict[str, Any]] = {}
semaphore: asyncio.Semaphore  # initialised in startup
purge_tasks: Dict[str, asyncio.Task] = {}
purge_index_path = STORAGE_ROOT / ".purge_index.json"
purge_index_lock = asyncio.Lock()
job_store_lock = threading.RLock()


def _job_store_connection() -> sqlite3.Connection:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(JOBS_DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_job_store() -> None:
    with _job_store_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                tender_id TEXT,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                processing_time_s REAL NOT NULL DEFAULT 0,
                errors_json TEXT NOT NULL DEFAULT '[]',
                logs_json TEXT NOT NULL DEFAULT '[]',
                documents_json TEXT NOT NULL DEFAULT '[]',
                full_text TEXT NOT NULL DEFAULT '',
                storage_path TEXT,
                purge_at TEXT,
                updated_at REAL NOT NULL,
                progress_pct INTEGER NOT NULL DEFAULT 0,
                documents_done INTEGER NOT NULL DEFAULT 0,
                documents_total INTEGER NOT NULL DEFAULT 0,
                current_document TEXT,
                current_step TEXT
            )
            """
        )
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        if "progress_pct" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN progress_pct INTEGER NOT NULL DEFAULT 0")
        if "documents_done" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN documents_done INTEGER NOT NULL DEFAULT 0")
        if "documents_total" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN documents_total INTEGER NOT NULL DEFAULT 0")
        if "current_document" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN current_document TEXT")
        if "current_step" not in existing_columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN current_step TEXT")
        conn.commit()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_load(raw: Optional[str], fallback: Any) -> Any:
    if raw in (None, ""):
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def _row_to_job(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "tender_id": row["tender_id"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "processing_time_s": row["processing_time_s"],
        "progress_pct": int(row["progress_pct"] or 0),
        "documents_done": int(row["documents_done"] or 0),
        "documents_total": int(row["documents_total"] or 0),
        "current_document": row["current_document"],
        "current_step": row["current_step"],
        "errors": _json_load(row["errors_json"], []),
        "logs": _json_load(row["logs_json"], []),
        "documents": _json_load(row["documents_json"], []),
        "full_text": row["full_text"] or "",
        "storage_path": row["storage_path"],
        "purge_at": row["purge_at"],
    }


def _job_store_upsert(job: Dict[str, Any]) -> None:
    payload = {
        "job_id": job["job_id"],
        "tender_id": job.get("tender_id"),
        "status": job.get("status", "pending"),
        "created_at": job.get("created_at", time.time()),
        "processing_time_s": job.get("processing_time_s", 0.0),
        "errors_json": _json_dump(job.get("errors", [])),
        "logs_json": _json_dump(job.get("logs", [])),
        "documents_json": _json_dump(job.get("documents", [])),
        "full_text": job.get("full_text", ""),
        "storage_path": job.get("storage_path"),
        "purge_at": job.get("purge_at"),
        "updated_at": job.get("updated_at", time.time()),
        "progress_pct": int(job.get("progress_pct", 0) or 0),
        "documents_done": int(job.get("documents_done", 0) or 0),
        "documents_total": int(job.get("documents_total", 0) or 0),
        "current_document": job.get("current_document"),
        "current_step": job.get("current_step"),
    }
    with job_store_lock:
        with _job_store_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, tender_id, status, created_at, processing_time_s,
                    errors_json, logs_json, documents_json, full_text,
                    storage_path, purge_at, updated_at,
                    progress_pct, documents_done, documents_total,
                    current_document, current_step
                ) VALUES (
                    :job_id, :tender_id, :status, :created_at, :processing_time_s,
                    :errors_json, :logs_json, :documents_json, :full_text,
                    :storage_path, :purge_at, :updated_at,
                    :progress_pct, :documents_done, :documents_total,
                    :current_document, :current_step
                )
                ON CONFLICT(job_id) DO UPDATE SET
                    tender_id=excluded.tender_id,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    processing_time_s=excluded.processing_time_s,
                    errors_json=excluded.errors_json,
                    logs_json=excluded.logs_json,
                    documents_json=excluded.documents_json,
                    full_text=excluded.full_text,
                    storage_path=excluded.storage_path,
                    purge_at=excluded.purge_at,
                    updated_at=excluded.updated_at,
                    progress_pct=excluded.progress_pct,
                    documents_done=excluded.documents_done,
                    documents_total=excluded.documents_total,
                    current_document=excluded.current_document,
                    current_step=excluded.current_step
                """,
                payload,
            )
            conn.commit()


def _job_store_get(job_id: str) -> Optional[Dict[str, Any]]:
    with _job_store_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def _job_store_list_recent(limit: int) -> List[Dict[str, Any]]:
    with _job_store_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_job(row) for row in rows]


def _job_store_list_active(limit: int = 20) -> List[Dict[str, Any]]:
    with _job_store_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status IN ('pending', 'processing', 'error') ORDER BY updated_at DESC, job_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_job(row) for row in rows]


def _job_store_queue_counts() -> Dict[str, Any]:
    counts: Dict[str, Any] = {"pending": 0, "processing": 0, "done": 0, "max_workers": MAX_WORKERS, "slots_in_use": 0, "slots_free": MAX_WORKERS, "oldest_pending_age_s": 0.0}
    with _job_store_connection() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS total FROM jobs GROUP BY status"
        ).fetchall()
        oldest_pending = conn.execute(
            "SELECT MIN(created_at) AS oldest_pending FROM jobs WHERE status = 'pending'"
        ).fetchone()
    for row in rows:
        status = row["status"]
        total = int(row["total"])
        if status in ("done", "error"):
            counts["done"] += total
        elif status in counts:
            counts[status] = total
    counts["slots_in_use"] = min(MAX_WORKERS, int(counts["processing"]))
    counts["slots_free"] = max(0, MAX_WORKERS - int(counts["slots_in_use"]))
    oldest_created = oldest_pending["oldest_pending"] if oldest_pending else None
    if oldest_created:
        counts["oldest_pending_age_s"] = round(max(0.0, time.time() - float(oldest_created)), 2)
    return counts


def _job_store_update(job_id: str, **updates: Any) -> Dict[str, Any]:
    with job_store_lock:
        current = _job_store_get(job_id)
        if current is None:
            raise KeyError(job_id)
        current.update(updates)
        current['updated_at'] = updates.get('updated_at', time.time())
        _job_store_upsert(current)
        return current


def _append_job_log(job_id: str, message: str) -> Dict[str, Any]:
    with job_store_lock:
        current = _job_store_get(job_id)
        if current is None:
            raise KeyError(job_id)
        logs = list(current.get("logs", []))
        logs.append(message)
        current["logs"] = logs
        current["updated_at"] = time.time()
        _job_store_upsert(current)
        return current


def _replace_job_documents(job_id: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _job_store_update(job_id, documents=documents)


def _compute_progress(documents_done: int, documents_total: int, status: str) -> int:
    if status == "done":
        return 100
    if documents_total <= 0:
        return 0
    return max(0, min(99, int((documents_done / max(1, documents_total)) * 100)))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class DocumentInput(BaseModel):
    filename: str
    url: Optional[str] = None
    path: Optional[str] = None


class ParseOptions(BaseModel):
    force_ocr: bool = False


class ParseRequest(BaseModel):
    tender_id: str
    documents: List[DocumentInput] = []
    manifest_url: Optional[str] = None
    storage_prefix: Optional[str] = None
    purge_after_days: int = 7
    options: Optional[ParseOptions] = None
    pipeline_job_id: Optional[int] = None
    correlation_id: Optional[str] = None
    celery_task_id: Optional[str] = None


class DownloadRequest(BaseModel):
    tender_id: str
    documents: List[DocumentInput]


class DownloadedFile(BaseModel):
    filename: str
    content_b64: str
    mime_type: str


class DownloadResponse(BaseModel):
    status: str
    files: List[DownloadedFile]
    errors: List[str]


class DocumentResult(BaseModel):
    filename: str
    type_detected: str
    method_used: str
    pages: int
    quality_score: float
    text: str
    error: Optional[str] = None
    logs: List[str] = []


class ParseResponse(BaseModel):
    tender_id: str
    status: str  # done | error | pending | processing
    job_id: str
    pipeline_job_id: Optional[int] = None
    correlation_id: Optional[str] = None
    celery_task_id: Optional[str] = None
    documents: List[DocumentResult] = []
    full_text: str = ""
    errors: List[str] = []
    logs: List[str] = []
    processing_time_s: float = 0.0


class EnrichmentResult(BaseModel):
    tender_id: str
    chunks_total: int
    chunks_ok: int
    processing_time_s: float
    resumo_ia: Optional[str] = None
    regras_licitacao: Optional[dict] = None
    itens: Optional[list] = None
    fornecedores_sugeridos: Optional[dict] = None
    raw_chunks: Optional[list] = None
    created_at: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Lobeira", version="2.0.0")


@app.on_event("startup")
async def startup():
    global semaphore
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    _init_job_store()
    await _restore_and_schedule_purges()
    logger.info(
        "Lobeira iniciado. mode=%s MAX_WORKERS=%d "
        "thresholds: pymupdf=%.2f paddleocr=%.2f tesseract=%.2f",
        PARSER_MODE,
        MAX_WORKERS,
        PYMUPDF_QUALITY_THRESHOLD,
        PADDLE_OCR_QUALITY_THRESHOLD,
        TESSERACT_QUALITY_THRESHOLD,
    )


_init_job_store()


# ---------------------------------------------------------------------------
# Health & queue
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        tess_ok = True
    except Exception:
        tess_ok = False
    queue = _job_store_queue_counts()
    return {
        "status": "ok",
        "tesseract": tess_ok,
        "parser_mode": PARSER_MODE,
        "thresholds": {
            "pymupdf": PYMUPDF_QUALITY_THRESHOLD,
            "paddleocr_vl": PADDLE_OCR_QUALITY_THRESHOLD,
            "tesseract": TESSERACT_QUALITY_THRESHOLD,
        },
        "paddle_ocr": {
            "url": PADDLE_OCR_URL,
            "model": PADDLE_OCR_MODEL,
            "timeout_s": PADDLE_OCR_TIMEOUT,
            "gpu_util": PADDLE_OCR_GPU_UTIL,
        },
        "max_workers": queue["max_workers"],
        "slots_in_use": queue["slots_in_use"],
        "slots_free": queue["slots_free"],
        "pending": queue["pending"],
        "processing": queue["processing"],
        "oldest_pending_age_s": queue["oldest_pending_age_s"],
    }


@app.get("/queue")
def queue_status():
    queue = _job_store_queue_counts()
    active_source_jobs = _job_store_list_active(20)
    active_jobs = [
        {
            "job_id": job.get("job_id"),
            "tender_id": job.get("tender_id"),
            "status": job.get("status"),
            "updated_at": job.get("updated_at"),
            "progress_pct": job.get("progress_pct", 0),
            "documents_done": job.get("documents_done", 0),
            "documents_total": job.get("documents_total", 0),
            "current_document": job.get("current_document"),
            "current_step": job.get("current_step"),
        }
        for job in active_source_jobs
    ]
    return {**queue, "parser_mode": PARSER_MODE, "active_jobs": active_jobs}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _job_store_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _normalize_system_log_entry(entry: Dict[str, Any], source: str) -> Dict[str, Any]:
    timestamp = (
        entry.get("timestamp")
        or entry.get("__REALTIME_TIMESTAMP")
        or entry.get("_SOURCE_REALTIME_TIMESTAMP")
        or entry.get("ts")
    )
    if timestamp and str(timestamp).isdigit():
        timestamp = datetime.fromtimestamp(int(str(timestamp)) / 1_000_000, tz=timezone.utc).isoformat()

    return {
        "timestamp": timestamp,
        "level": entry.get("level") or entry.get("PRIORITY") or entry.get("priority"),
        "logger": entry.get("logger") or entry.get("SYSLOG_IDENTIFIER") or entry.get("_COMM"),
        "pid": entry.get("pid") or entry.get("_PID"),
        "message": _stringify_system_log_message(entry.get("message") or entry.get("MESSAGE") or ""),
        "source": source,
    }


def _stringify_system_log_message(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, bytes):
        return message.decode("utf-8", errors="replace")
    if isinstance(message, (list, tuple)):
        return " ".join(part for part in (_stringify_system_log_message(item) for item in message) if part)
    if isinstance(message, dict):
        try:
            return json.dumps(message, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(message)
    return str(message)


def _is_noisy_system_log_message(message: Any) -> bool:
    text = _stringify_system_log_message(message)
    return any(pattern.search(text) for pattern in SYSTEM_LOG_ACCESS_PATTERNS)


def _filter_system_log_entries(
    entries: Iterable[Dict[str, Any]],
    limit: int,
    contains: Optional[str] = None,
    include_access_logs: bool = False,
) -> List[Dict[str, Any]]:
    contains_normalized = (contains or "").strip().lower()
    filtered: List[Dict[str, Any]] = []
    for raw_entry in entries:
        entry = _normalize_system_log_entry(raw_entry, raw_entry.get("source", "unknown"))
        message = _stringify_system_log_message(entry.get("message") or "")
        entry["message"] = message
        if not include_access_logs and _is_noisy_system_log_message(message):
            continue
        if contains_normalized and contains_normalized not in message.lower():
            continue
        filtered.append(entry)

    filtered.sort(key=lambda item: item.get("timestamp") or "", reverse=True)
    return filtered[:limit]


def _read_system_logs_from_buffer(
    limit: int,
    contains: Optional[str] = None,
    include_access_logs: bool = False,
) -> List[Dict[str, Any]]:
    with _system_log_buffer_lock:
        snapshot = [dict(entry) for entry in _system_log_buffer]
    return _filter_system_log_entries(snapshot, limit=limit, contains=contains, include_access_logs=include_access_logs)


def _read_system_logs_from_file(
    log_path: str,
    limit: int,
    contains: Optional[str] = None,
    include_access_logs: bool = False,
) -> List[Dict[str, Any]]:
    path = Path(log_path)
    if not path.exists() or not path.is_file():
        return []

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries = [
        {"timestamp": None, "level": None, "logger": None, "pid": None, "message": line, "source": "file"}
        for line in lines[-max(limit * 5, limit):]
        if line.strip()
    ]
    return _filter_system_log_entries(entries, limit=limit, contains=contains, include_access_logs=include_access_logs)


def _journalctl_args(limit: int, since: Optional[str]) -> List[str]:
    args = ["journalctl", "--no-pager", "-o", "json", "-n", str(max(limit * 5, limit))]
    if since:
        args.extend(["--since", since])
    if SYSTEMD_INVOCATION_ID:
        args.append(f"_SYSTEMD_INVOCATION_ID={SYSTEMD_INVOCATION_ID}")
    elif SYSTEMD_UNIT:
        args.extend(["-u", SYSTEMD_UNIT])
    else:
        args.append("SYSLOG_IDENTIFIER=parser-monstro")
    return args


def _read_system_logs_from_journal(
    limit: int,
    since: Optional[str] = None,
    contains: Optional[str] = None,
    include_access_logs: bool = False,
) -> Dict[str, Any]:
    args = _journalctl_args(limit, since)
    completed = subprocess.run(args, capture_output=True, text=True, check=False, timeout=10)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "journalctl failed").strip())

    entries = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        item["source"] = "journalctl"
        entries.append(item)

    return {
        "backend": "journalctl",
        "entries": _filter_system_log_entries(entries, limit=limit, contains=contains, include_access_logs=include_access_logs),
    }


def _get_system_logs(
    limit: int,
    since: Optional[str] = None,
    contains: Optional[str] = None,
    include_access_logs: bool = False,
) -> Dict[str, Any]:
    warnings: List[str] = []
    backend = None
    entries: List[Dict[str, Any]] = []

    try:
        result = _read_system_logs_from_journal(limit=limit, since=since, contains=contains, include_access_logs=include_access_logs)
        backend = result["backend"]
        entries = result["entries"]
    except Exception as exc:
        warnings.append(f"journalctl indisponível ou sem permissão: {exc}")

    if not entries and SYSTEM_LOG_FILE_PATH:
        file_entries = _read_system_logs_from_file(SYSTEM_LOG_FILE_PATH, limit=limit, contains=contains, include_access_logs=include_access_logs)
        if file_entries:
            backend = "file"
            entries = file_entries

    if not entries:
        buffer_entries = _read_system_logs_from_buffer(limit=limit, contains=contains, include_access_logs=include_access_logs)
        if buffer_entries:
            backend = "in_memory"
            entries = buffer_entries

    return {
        "backend": backend or "none",
        "filters": {
            "limit": limit,
            "since": since,
            "contains": contains,
            "include_access_logs": include_access_logs,
            "systemd_unit": SYSTEMD_UNIT or None,
            "invocation_id": SYSTEMD_INVOCATION_ID or None,
            "fallback_file_path": SYSTEM_LOG_FILE_PATH or None,
        },
        "count": len(entries),
        "items": entries,
        "warnings": warnings,
    }


def _build_job_logs_payload(job_id: str, include_documents: bool = False) -> Dict[str, Any]:
    job = _job_store_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    payload: Dict[str, Any] = {
        "job_id": job["job_id"],
        "tender_id": job.get("tender_id"),
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "processing_time_s": job.get("processing_time_s", 0.0),
        "progress_pct": job.get("progress_pct", 0),
        "documents_done": job.get("documents_done", 0),
        "documents_total": job.get("documents_total", 0),
        "current_document": job.get("current_document"),
        "current_step": job.get("current_step"),
        "errors": job.get("errors", []),
        "logs": job.get("logs", []),
    }
    if include_documents:
        payload["documents"] = [
            {
                "filename": doc.get("filename"),
                "method_used": doc.get("method_used"),
                "error": doc.get("error"),
                "logs": doc.get("logs", []),
            }
            for doc in job.get("documents", [])
        ]
    return payload


@app.get("/logs/recent")
def get_recent_logs(limit: int = Query(default=20, ge=1, le=100)):
    selected_jobs = _job_store_list_recent(limit)
    return {
        "count": len(selected_jobs),
        "items": [_build_job_logs_payload(job["job_id"]) for job in selected_jobs],
    }


@app.get("/logs/job/{job_id}")
def get_job_logs(job_id: str):
    return _build_job_logs_payload(job_id, include_documents=True)


@app.get("/logs/system/recent")
def get_recent_system_logs(
    limit: int = Query(default=50, ge=1, le=200),
    since: Optional[str] = Query(default=None),
    contains: Optional[str] = Query(default=None),
    include_access_logs: bool = Query(default=False),
):
    return _get_system_logs(
        limit=limit,
        since=since,
        contains=contains,
        include_access_logs=include_access_logs,
    )


@app.get("/storage/{tender_id}")
def list_storage(tender_id: str):
    target_dir = STORAGE_ROOT / tender_id
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail="Storage not found for tender")
    files = [str(p.relative_to(target_dir)) for p in target_dir.rglob("*") if p.is_file()]
    return {
        "tender_id": tender_id,
        "storage_path": str(target_dir),
        "files": sorted(files),
        "count": len(files),
    }


@app.post("/storage/{tender_id}/enrichment")
def save_enrichment(tender_id: str, payload: EnrichmentResult):
    if payload.tender_id != tender_id:
        raise HTTPException(status_code=400, detail="tender_id mismatch between path and payload")

    target_dir = STORAGE_ROOT / tender_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "enrichment.json"
    target_file.write_text(
        json.dumps(payload.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "tender_id": tender_id,
        "saved_path": str(target_file),
    }


@app.get("/storage/{tender_id}/enrichment")
def get_enrichment(tender_id: str):
    target_file = STORAGE_ROOT / tender_id / "enrichment.json"
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="Enrichment not found for tender")

    try:
        return json.loads(target_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read enrichment file: {exc}")


@app.delete("/storage/{tender_id}")
async def delete_storage(tender_id: str):
    deleted = await _purge_tender_storage(tender_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Storage not found for tender")
    return {"tender_id": tender_id, "deleted": True}



def _load_manifest_documents(req: ParseRequest) -> List[DocumentInput]:
    if req.documents:
        return req.documents
    if not req.manifest_url:
        return []
    resp = requests.get(req.manifest_url, timeout=60)
    resp.raise_for_status()
    manifest = resp.json()
    docs: List[DocumentInput] = []
    for item in manifest.get("prepared_files") or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        path = item.get("path")
        filename = Path(str(item.get("filename") or item.get("storage_key") or "documento")).name
        if not url and not path:
            continue
        docs.append(DocumentInput(filename=filename, url=url, path=path))
    return docs


# ---------------------------------------------------------------------------
# Main parse endpoint
# ---------------------------------------------------------------------------
@app.post("/parse", response_model=ParseResponse, status_code=202)
async def parse_documents(req: ParseRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    documents = _load_manifest_documents(req)
    if not documents:
        raise HTTPException(status_code=400, detail="No documents resolved for parse request")
    purge_at = datetime.now(timezone.utc) + timedelta(days=max(1, req.purge_after_days))
    storage_path = STORAGE_ROOT / req.tender_id
    _job_store_upsert(
        {
            "job_id": job_id,
            "tender_id": req.tender_id,
            "status": "pending",
            "pipeline_job_id": req.pipeline_job_id,
            "correlation_id": req.correlation_id,
            "celery_task_id": req.celery_task_id,
            "documents": [],
            "full_text": "",
            "errors": [],
            "logs": [f"[{datetime.now(timezone.utc).isoformat()}] job criado e aguardando worker (pipeline_job_id={req.pipeline_job_id}, correlation_id={req.correlation_id}, celery_task_id={req.celery_task_id})"],
            "processing_time_s": 0.0,
            "progress_pct": 0,
            "documents_done": 0,
            "documents_total": len(documents),
            "current_document": None,
            "current_step": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "storage_path": str(storage_path),
            "purge_at": purge_at.isoformat(),
        }
    )
    background_tasks.add_task(_process_job, job_id, req, documents)
    logger.info("parser_job_queued job_id=%s tender_id=%s pipeline_job_id=%s correlation_id=%s celery_task_id=%s docs_total=%s", job_id, req.tender_id, req.pipeline_job_id, req.correlation_id, req.celery_task_id, len(documents))
    return ParseResponse(
        tender_id=req.tender_id,
        status="pending",
        job_id=job_id,
        pipeline_job_id=req.pipeline_job_id,
        correlation_id=req.correlation_id,
        celery_task_id=req.celery_task_id,
    )


@app.post("/download", response_model=DownloadResponse)
async def download_documents(req: DownloadRequest):
    tender_root = STORAGE_ROOT / req.tender_id
    raw_dir = tender_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files: List[DownloadedFile] = []
    errors: List[str] = []

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for doc in req.documents:
            url = (doc.url or "").strip()
            filename = Path(doc.filename or "documento").name or "documento"
            if not url:
                errors.append(f"Documento sem URL: {filename}")
                continue

            try:
                resp = await client.get(url)
                resp.raise_for_status()
                content = resp.content
            except Exception as exc:
                errors.append(f"Falha ao baixar {filename} ({url}): {exc}")
                continue

            raw_path = raw_dir / filename
            try:
                raw_path.write_bytes(content)
            except Exception as exc:
                errors.append(f"Falha ao salvar raw {filename}: {exc}")
                continue

            files_to_send: List[Path] = []
            if content[:2] == b"PK":
                try:
                    extracted = _extract_zip(raw_path, raw_dir)
                    files_to_send = [p for p in extracted if p.is_file()]
                except Exception as exc:
                    errors.append(f"Falha ao extrair ZIP {filename}: {exc}")
                    continue
            else:
                files_to_send = [raw_path]

            for fpath in files_to_send:
                try:
                    file_bytes = fpath.read_bytes()
                    mime_type = magic.from_buffer(file_bytes, mime=True) or "application/octet-stream"
                    files.append(
                        DownloadedFile(
                            filename=fpath.name,
                            content_b64=base64.b64encode(file_bytes).decode("ascii"),
                            mime_type=mime_type,
                        )
                    )
                except Exception as exc:
                    errors.append(f"Falha ao preparar arquivo {fpath.name}: {exc}")

    return DownloadResponse(status="ok", files=files, errors=errors)


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------
async def _process_job(job_id: str, req: ParseRequest, documents: List[DocumentInput]):
    async with semaphore:
        _job_store_update(job_id, status="processing", progress_pct=0, documents_done=0, documents_total=len(documents), current_document=None, current_step="starting", updated_at=time.time())
        _append_job_log(job_id, f"[{datetime.now(timezone.utc).isoformat()}] worker iniciou processamento")
        logger.info("parser_job_started job_id=%s tender_id=%s pipeline_job_id=%s correlation_id=%s celery_task_id=%s documentos=%d", job_id, req.tender_id, req.pipeline_job_id, req.correlation_id, req.celery_task_id, len(documents))
        await asyncio.to_thread(_process_job_sync, job_id, req, documents)
        job = _job_store_get(job_id)
        purge_at_iso = job.get("purge_at") if job else None
        if purge_at_iso:
            purge_at = datetime.fromisoformat(purge_at_iso)
            if purge_at.tzinfo is None:
                purge_at = purge_at.replace(tzinfo=timezone.utc)
            await _upsert_purge_schedule(req.tender_id, purge_at)


def _process_job_sync(job_id: str, req: ParseRequest, documents: List[DocumentInput]):
    t0 = time.time()
    doc_results: List[Dict[str, Any]] = []
    errors: List[str] = []
    use_easyocr = False  # EasyOCR removido — não utilizado
    force_ocr = bool(req.options and req.options.force_ocr)
    target_dir = STORAGE_ROOT / req.tender_id
    target_dir.mkdir(parents=True, exist_ok=True)

    documents_total = len(documents)
    documents_done = 0
    try:
        for index, doc in enumerate(documents, start=1):
            _job_store_update(job_id, documents_done=documents_done, documents_total=documents_total, progress_pct=_compute_progress(documents_done, documents_total, 'processing'), current_document=doc.filename, current_step='starting_document', updated_at=time.time())
            _append_job_log(job_id, f"[{datetime.now(timezone.utc).isoformat()}] iniciando documento {index}/{len(documents)}: {doc.filename}")
            logger.info("Job %s processando documento %s/%s: %s", job_id, index, len(documents), doc.filename)
            try:
                results = asyncio.run(_handle_document(doc, target_dir, use_easyocr, force_ocr, job_id=job_id))
                doc_results.extend(results)
                documents_done += 1
                _replace_job_documents(job_id, doc_results)
                _job_store_update(job_id, documents_done=documents_done, documents_total=documents_total, progress_pct=_compute_progress(documents_done, documents_total, 'processing'), current_document=doc.filename, current_step='document_done', updated_at=time.time())
                _append_job_log(job_id, f"[{datetime.now(timezone.utc).isoformat()}] documento concluído: {doc.filename} ({len(results)} arquivo(s) gerado(s))")
                logger.info("Job %s concluiu documento %s com %d resultado(s)", job_id, doc.filename, len(results))
            except Exception as exc:
                logger.exception("Erro ao processar %s", doc.filename)
                err = f"{doc.filename}: {exc}"
                errors.append(err)
                _job_store_update(job_id, errors=errors, documents_done=documents_done, documents_total=documents_total, progress_pct=_compute_progress(documents_done, documents_total, 'processing'), current_document=doc.filename, current_step='document_error', updated_at=time.time())
                _append_job_log(job_id, f"[erro] {err}")

        full_text = "\n\n".join(r["text"] for r in doc_results if r.get("text"))
        parsed_dir = target_dir / "parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        if full_text:
            (parsed_dir / "full_text.txt").write_text(full_text, encoding="utf-8")
        for r in doc_results:
            if r.get("text") and r.get("filename"):
                (parsed_dir / (r["filename"] + ".txt")).write_text(r["text"], encoding="utf-8")

        all_logs = []
        for r in doc_results:
            fname = r.get("filename", "?")
            for entry in r.get("logs", []):
                all_logs.append(f"[{fname}] {entry}")
        for err in errors:
            all_logs.append(f"[erro] {err}")

        existing = _job_store_get(job_id) or {}
        merged_logs = list(existing.get("logs", [])) + all_logs
        status = "done" if doc_results else "error"
        processing_time = round(time.time() - t0, 2)
        purge_at = datetime.now(timezone.utc) + timedelta(days=max(1, req.purge_after_days))

        _job_store_update(
            job_id,
            status=status,
            documents=doc_results,
            full_text=full_text,
            errors=errors,
            logs=merged_logs,
            processing_time_s=processing_time,
            storage_path=str(target_dir),
            purge_at=purge_at.isoformat(),
            documents_done=documents_done,
            documents_total=documents_total,
            progress_pct=_compute_progress(documents_done, documents_total, status),
            current_document=None,
            current_step='finished',
            updated_at=time.time(),
        )
        _append_job_log(job_id, f"[{datetime.now(timezone.utc).isoformat()}] job concluído com status={status} em {processing_time:.2f}s")
        logger.info("parser_job_finished job_id=%s tender_id=%s pipeline_job_id=%s correlation_id=%s celery_task_id=%s status=%s processing_time_s=%.2f docs=%d errors=%d", job_id, req.tender_id, req.pipeline_job_id, req.correlation_id, req.celery_task_id, status, processing_time, len(doc_results), len(errors))
    except Exception as exc:
        logger.exception("Falha fatal no job %s", job_id)
        processing_time = round(time.time() - t0, 2)
        current = _job_store_get(job_id) or {}
        errors = list(current.get("errors", [])) + [str(exc)]
        logs = list(current.get("logs", [])) + [f"[erro] falha fatal do job: {exc}"]
        _job_store_update(
            job_id,
            status="error",
            errors=errors,
            logs=logs,
            processing_time_s=processing_time,
            storage_path=str(target_dir),
            documents_done=documents_done if 'documents_done' in locals() else 0,
            documents_total=documents_total if 'documents_total' in locals() else len(documents),
            progress_pct=_compute_progress(documents_done if 'documents_done' in locals() else 0, documents_total if 'documents_total' in locals() else len(documents), 'error'),
            current_document=None,
            current_step='fatal_error',
            updated_at=time.time(),
        )


async def _handle_document(doc: DocumentInput, tmpdir: Path, use_easyocr: bool, force_ocr: bool, job_id: Optional[str] = None) -> List[Dict]:
    """Download + decompress + parse one document. Returns list of DocumentResult dicts.
    use_easyocr mantido por compatibilidade de assinatura; sem efeito."""
    safe_name = Path(doc.filename).name or f"doc_{uuid.uuid4().hex}"
    dest = tmpdir / safe_name
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(doc.url)
        resp.raise_for_status()
    dest.write_bytes(resp.content)

    mime = magic.from_file(str(dest), mime=True)
    logger.debug("%s → mime=%s", doc.filename, mime)

    files_to_parse: List[Path] = []
    if mime in ("application/zip", "application/x-zip-compressed") or doc.filename.lower().endswith(".zip"):
        files_to_parse = _extract_zip(dest, tmpdir)
    elif mime in ("application/x-rar-compressed", "application/vnd.rar") or doc.filename.lower().endswith(".rar"):
        files_to_parse = _extract_rar(dest, tmpdir)
    elif mime == "application/x-7z-compressed" or doc.filename.lower().endswith(".7z"):
        files_to_parse = _extract_7z(dest, tmpdir)
    else:
        files_to_parse = [dest]

    results = []
    for fpath in files_to_parse:
        result = _parse_file(fpath, use_easyocr, force_ocr, job_id=job_id)
        results.append(result)
    return results


def _extract_zip(path: Path, dest: Path) -> List[Path]:
    out = dest / f"_zip_{path.stem}"
    out.mkdir(exist_ok=True)

    limits = ZipExtractionLimits()
    extraction = extract_zip_recursive(
        archive_name=path.name,
        archive_bytes=path.read_bytes(),
        limits=limits,
        logger=logger,
        log_prefix=f"parser-zip:{path.name}",
    )

    extracted_paths = write_extracted_files(out, extraction.files, logger=logger)
    if extraction.warnings:
        logger.warning(
            "[parser-zip:%s] extração parcial com avisos warnings=%s",
            path.name,
            len(extraction.warnings),
        )

    return extracted_paths


def _extract_rar(path: Path, dest: Path) -> List[Path]:
    import rarfile
    out = dest / f"_rar_{path.stem}"
    out.mkdir(exist_ok=True)
    with rarfile.RarFile(str(path)) as rf:
        rf.extractall(str(out))
    return [p for p in out.rglob("*") if p.is_file()]


def _extract_7z(path: Path, dest: Path) -> List[Path]:
    import py7zr
    out = dest / f"_7z_{path.stem}"
    out.mkdir(exist_ok=True)
    with py7zr.SevenZipFile(str(path), mode="r") as zf:
        zf.extractall(path=str(out))
    return [p for p in out.rglob("*") if p.is_file()]


async def _restore_and_schedule_purges() -> None:
    if not purge_index_path.exists():
        return
    try:
        raw = json.loads(purge_index_path.read_text())
    except Exception:
        logger.warning("Falha ao ler índice de purge; ignorando")
        return

    now = datetime.now(timezone.utc)
    changed = False
    for tender_id, purge_at_iso in raw.items():
        try:
            purge_at = datetime.fromisoformat(purge_at_iso)
            if purge_at.tzinfo is None:
                purge_at = purge_at.replace(tzinfo=timezone.utc)
        except Exception:
            changed = True
            continue

        if purge_at <= now:
            await _purge_tender_storage(tender_id)
            changed = True
            continue
        _schedule_purge_task(tender_id, purge_at)

    if changed:
        await _save_purge_index(await _load_purge_index())


async def _load_purge_index() -> Dict[str, str]:
    async with purge_index_lock:
        if not purge_index_path.exists():
            return {}
        try:
            return json.loads(purge_index_path.read_text())
        except Exception:
            return {}


async def _save_purge_index(data: Dict[str, str]) -> None:
    async with purge_index_lock:
        purge_index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _schedule_purge_task(tender_id: str, purge_at: datetime) -> None:
    existing = purge_tasks.get(tender_id)
    if existing and not existing.done():
        existing.cancel()
    purge_tasks[tender_id] = asyncio.create_task(_purge_after_delay(tender_id, purge_at))


async def _upsert_purge_schedule(tender_id: str, purge_at: datetime) -> None:
    data = await _load_purge_index()
    data[tender_id] = purge_at.isoformat()
    await _save_purge_index(data)
    _schedule_purge_task(tender_id, purge_at)


async def _purge_after_delay(tender_id: str, purge_at: datetime) -> None:
    delay = max(0, (purge_at - datetime.now(timezone.utc)).total_seconds())
    await asyncio.sleep(delay)
    await _purge_tender_storage(tender_id)


async def _purge_tender_storage(tender_id: str) -> bool:
    target_dir = STORAGE_ROOT / tender_id
    existed = target_dir.exists() and target_dir.is_dir()
    if existed:
        import shutil
        shutil.rmtree(target_dir, ignore_errors=True)

    data = await _load_purge_index()
    if tender_id in data:
        data.pop(tender_id, None)
        await _save_purge_index(data)

    task = purge_tasks.pop(tender_id, None)
    if task and not task.done():
        task.cancel()
    return existed


def _parse_file(path: Path, use_easyocr: bool, force_ocr: bool, job_id: Optional[str] = None) -> Dict:
    """Parse a single file with fallback chain."""
    mime = magic.from_file(str(path), mime=True)
    filename = path.name

    if "pdf" in mime:
        return _parse_pdf(path, use_easyocr, force_ocr, job_id=job_id)
    elif "word" in mime or "officedocument" in mime or path.suffix.lower() in (".doc", ".docx"):
        return _parse_docx(path)
    elif "html" in mime or path.suffix.lower() in (".html", ".htm"):
        return _parse_html(path)
    elif "text" in mime or path.suffix.lower() == ".txt":
        text = path.read_text(errors="replace")
        return _make_result(filename, "plain_text", "read", 1, 1.0, text)
    elif mime.startswith("image/"):
        return _parse_image_ocr(path)
    else:
        return _make_result(filename, mime, "unsupported", 0, 0.0, "", error=f"Tipo não suportado: {mime}")


def _parse_pdf(path: Path, use_easyocr: bool, force_ocr: bool, job_id: Optional[str] = None) -> Dict:
    """Fallback chain: pymupdf → paddleocr-vl → tesseract → revisão manual."""
    filename = path.name
    text = ""
    method = ""
    pages = 0
    quality = 0.0
    logs: List[str] = []
    review_dir = REVIEW_ROOT / (job_id or uuid.uuid4().hex) / path.stem

    try:
        import fitz
        doc = fitz.open(str(path))
        pages = doc.page_count
        parts = [(doc.load_page(i).get_text("text") or "") for i in range(pages)]
        doc.close()
        raw = "\n".join(parts).strip()
        text = _normalize_text(raw)
        if text:
            method = "pymupdf"
            quality = _quality_score(text, pages)
            chars_per_page = len(text) / max(1, pages)
            if chars_per_page < MIN_CHARS_PER_PAGE_NATIVE:
                quality = min(quality, PYMUPDF_QUALITY_THRESHOLD - 0.01)
            logs.append(f"pymupdf: {len(text)} chars, {pages}p, score={quality:.2f}")
        else:
            logs.append("pymupdf: sem texto nativo")
    except Exception as e:
        logs.append(f"pymupdf: falhou ({e})")

    if not force_ocr and quality >= PYMUPDF_QUALITY_THRESHOLD:
        logs.append(f"pymupdf: suficiente (score={quality:.2f} >= {PYMUPDF_QUALITY_THRESHOLD}), parando chain")
        return _make_result(filename, "pdf_native", method, pages, quality, text, logs=logs)

    logs.append(f"paddleocr_vl: tentando (score atual={quality:.2f}, timeout={PADDLE_OCR_TIMEOUT}s/página)")
    try:
        paddle_result = PaddleOCRVLParser(
            base_url=PADDLE_OCR_URL,
            model=PADDLE_OCR_MODEL,
            timeout_s=PADDLE_OCR_TIMEOUT,
            image_dpi=PADDLE_OCR_IMAGE_DPI,
        ).parse_pdf(path, review_dir=review_dir)
        paddle_text = _normalize_text(paddle_result.get("text", ""))
        paddle_pages = int(paddle_result.get("pages") or pages or 0)
        paddle_quality = float(paddle_result.get("quality") or 0.0)
        logs.extend(paddle_result.get("logs", []))
        if paddle_text:
            if force_ocr or paddle_quality >= quality:
                text = paddle_text
                pages = paddle_pages
                quality = paddle_quality
                method = "paddleocr_vl"
        else:
            logs.append("paddleocr_vl: resultado vazio")
    except Exception as e:
        logs.append(f"paddleocr_vl: falhou ({str(e)[:300]})")

    if quality >= PADDLE_OCR_QUALITY_THRESHOLD:
        logs.append(f"paddleocr_vl: suficiente (score={quality:.2f} >= {PADDLE_OCR_QUALITY_THRESHOLD}), parando chain")
        return _make_result(filename, "pdf_scanned", method, pages, quality, text, logs=logs)

    logs.append(f"tesseract: tentando (score atual={quality:.2f})")
    try:
        text_ocr, pages_ocr, _, failed_pages = _pdf_ocr_tesseract(path, review_dir=review_dir)
        ocr_text = _normalize_text(text_ocr)
        if ocr_text:
            ocr_pages = pages_ocr or max(1, pages)
            ocr_quality = _quality_score(ocr_text, ocr_pages)
            logs.append(f"tesseract: {len(ocr_text)} chars, {ocr_pages}p, score={ocr_quality:.2f}")
            if force_ocr or not text or ocr_quality >= quality:
                text = ocr_text
                pages = ocr_pages
                quality = ocr_quality
                method = "tesseract"
        else:
            logs.append("tesseract: resultado vazio")
        if failed_pages:
            logs.append(f"tesseract: páginas salvas para revisão manual={failed_pages}")
    except Exception as e:
        logs.append(f"tesseract: falhou ({e})")

    if quality >= TESSERACT_QUALITY_THRESHOLD:
        logs.append(f"tesseract: suficiente (score={quality:.2f} >= {TESSERACT_QUALITY_THRESHOLD}), parando chain")
        return _make_result(filename, "pdf_scanned", method, pages, quality, text, logs=logs)

    logs.append(f"review: falha final do chain para {filename}; páginas problemáticas em {review_dir}")
    logger.error("parser_review_required file=%s job_id=%s review_dir=%s score=%.2f method=%s", filename, job_id, review_dir, quality, method or "failed")
    return _make_result(
        filename,
        "pdf_scanned",
        method or "failed",
        pages,
        quality,
        text,
        error=f"OCR abaixo do threshold mínimo; revisar páginas em {review_dir}",
        logs=logs,
    )


def _pdf_ocr_tesseract(path: Path, review_dir: Optional[Path] = None) -> tuple:
    """Converte PDF para imagens e extrai texto via Tesseract."""
    import fitz
    import pytesseract
    from PIL import Image

    review_dir = review_dir or REVIEW_ROOT / f"adhoc-{uuid.uuid4().hex}"
    review_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(path))
    parts = []
    failed_pages: List[int] = []
    page_count = doc.page_count
    for i in range(page_count):
        page_number = i + 1
        pix = doc.load_page(i).get_pixmap(dpi=200, alpha=False)
        image_bytes = pix.tobytes("png")
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_text = _normalize_text(pytesseract.image_to_string(img, lang="por"))
        if page_text:
            parts.append(page_text)
        else:
            failed_pages.append(page_number)
            (review_dir / f"page-{page_number:04d}.png").write_bytes(image_bytes)
    doc.close()

    return "\n".join(parts).strip(), page_count, "tesseract", failed_pages


def _parse_docx(path: Path) -> Dict:
    filename = path.name
    try:
        from docx import Document
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        pages = max(1, len(text) // 3000)
        return _make_result(filename, "docx", "python-docx", pages, _quality_score(text, pages), text)
    except Exception as e:
        return _make_result(filename, "docx", "failed", 0, 0.0, "", error=str(e))


def _parse_html(path: Path) -> Dict:
    filename = path.name
    try:
        from bs4 import BeautifulSoup
        html = path.read_text(errors="replace")
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n")
        return _make_result(filename, "html", "beautifulsoup4", 1, _quality_score(text, 1), text)
    except Exception as e:
        return _make_result(filename, "html", "failed", 0, 0.0, "", error=str(e))


def _parse_image_ocr(path: Path) -> Dict:
    filename = path.name
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(str(path))
        text = pytesseract.image_to_string(img, lang="por")
        return _make_result(filename, "image", "tesseract", 1, _quality_score(text, 1), text)
    except Exception as e:
        return _make_result(filename, "image", "failed", 0, 0.0, "", error=str(e))


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    txt = unicodedata.normalize("NFKC", text)
    txt = txt.replace("\x00", " ")

    if CLEAN_OCR_NOISE:
        cleaned_lines = []
        for line in txt.splitlines():
            s = line.strip()
            if not s:
                continue
            non_word_ratio = len(re.findall(r"[^\w\s]", s)) / max(1, len(s))
            vowels = len(re.findall(r"[aeiouáéíóúâêôãõàü]", s, flags=re.IGNORECASE))
            if len(s) >= 24 and vowels == 0 and non_word_ratio > 0.25:
                continue
            cleaned_lines.append(s)
        txt = "\n".join(cleaned_lines)

    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _quality_score(text: str, pages: int) -> float:
    if not text or not pages:
        return 0.0
    words = len(text.split())
    words_per_page = words / max(pages, 1)
    score = min(1.0, words_per_page / 250)
    return round(score, 2)


def _make_result(filename, type_detected, method, pages, quality, text, error=None, logs=None) -> Dict:
    normalized_text = _normalize_text(text)
    if not quality and normalized_text and pages:
        quality = _quality_score(normalized_text, pages)
    return {
        "filename": filename,
        "type_detected": type_detected,
        "method_used": method,
        "pages": max(pages, 1) if normalized_text else pages,
        "quality_score": quality,
        "text": normalized_text,
        "error": error,
        "logs": logs or [],
    }
