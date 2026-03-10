import asyncio
import importlib
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

sys.modules.setdefault("httpx", types.SimpleNamespace())
sys.modules.setdefault(
    "magic",
    types.SimpleNamespace(
        from_file=lambda *args, **kwargs: "application/pdf",
        from_buffer=lambda *args, **kwargs: "application/pdf",
    ),
)


class _HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _FastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return lambda fn: fn

    def post(self, *args, **kwargs):
        return lambda fn: fn

    def delete(self, *args, **kwargs):
        return lambda fn: fn

    def on_event(self, *args, **kwargs):
        return lambda fn: fn


class _BackgroundTasks:
    def add_task(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _Query(default=None, **kwargs):
    return default


class _BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self):
        return self.__dict__.copy()


sys.modules.setdefault(
    "fastapi",
    types.SimpleNamespace(
        BackgroundTasks=_BackgroundTasks,
        FastAPI=_FastAPI,
        HTTPException=_HTTPException,
        Query=_Query,
    ),
)
sys.modules.setdefault("pydantic", types.SimpleNamespace(BaseModel=_BaseModel))


class LogsEndpointsTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="parser-monstro-tests-")
        os.environ["STORAGE_ROOT"] = self.tmpdir
        sys.modules.pop("main", None)
        import main  # noqa: F401
        self.main = importlib.import_module("main")

    def tearDown(self):
        sys.modules.pop("main", None)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _seed_jobs(self):
        self.main._job_store_upsert(
            {
                "job_id": "job-older",
                "tender_id": "tender-1",
                "status": "done",
                "created_at": 100.0,
                "processing_time_s": 1.25,
                "errors": [],
                "logs": ["[doc-a.pdf] pymupdf: ok"],
                "documents": [
                    {
                        "filename": "doc-a.pdf",
                        "method_used": "pymupdf",
                        "error": None,
                        "logs": ["pymupdf: ok"],
                    }
                ],
            }
        )
        self.main._job_store_upsert(
            {
                "job_id": "job-newer",
                "tender_id": "tender-2",
                "status": "error",
                "created_at": 200.0,
                "processing_time_s": 0.75,
                "errors": ["Falha ao baixar arquivo"],
                "logs": ["[erro] Falha ao baixar arquivo"],
                "documents": [],
            }
        )

    def test_get_recent_logs_returns_newest_jobs_first_and_respects_limit(self):
        self._seed_jobs()

        payload = self.main.get_recent_logs(limit=1)

        self.assertEqual(payload["count"], 1)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["job_id"], "job-newer")
        self.assertNotIn("documents", payload["items"][0])
        self.assertEqual(payload["items"][0]["logs"], ["[erro] Falha ao baixar arquivo"])

    def test_get_job_logs_returns_document_level_logs(self):
        self._seed_jobs()

        payload = self.main.get_job_logs("job-older")

        self.assertEqual(payload["job_id"], "job-older")
        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["logs"], ["[doc-a.pdf] pymupdf: ok"])
        self.assertEqual(len(payload["documents"]), 1)
        self.assertEqual(payload["documents"][0]["filename"], "doc-a.pdf")
        self.assertEqual(payload["documents"][0]["logs"], ["pymupdf: ok"])

    def test_get_job_logs_returns_404_for_missing_job(self):
        with self.assertRaises(self.main.HTTPException) as ctx:
            self.main.get_job_logs("job-missing")

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, "Job not found")

    def test_queue_uses_sqlite_store_not_in_memory_jobs(self):
        self._seed_jobs()
        self.main.jobs["ghost"] = {"status": "processing"}

        payload = self.main.queue_status()

        self.assertEqual(payload, {"pending": 0, "processing": 0, "done": 2})

    def test_active_job_logs_and_queue_remain_visible_during_processing(self):
        req = self.main.ParseRequest(
            tender_id="tender-live",
            documents=[self.main.DocumentInput(url="http://example/doc.pdf", filename="doc.pdf")],
        )
        job_id = "job-live"
        self.main._job_store_upsert(
            {
                "job_id": job_id,
                "tender_id": req.tender_id,
                "status": "pending",
                "created_at": time.time(),
                "processing_time_s": 0.0,
                "errors": [],
                "logs": ["queued"],
                "documents": [],
                "full_text": "",
                "storage_path": str(Path(self.tmpdir) / req.tender_id),
            }
        )

        started = threading.Event()
        release = threading.Event()
        original_handle_document = self.main._handle_document
        original_to_thread = self.main.asyncio.to_thread

        async def fake_handle_document(doc, target_dir, use_easyocr, force_ocr):
            started.set()
            release.wait(timeout=5)
            return [
                self.main._make_result(
                    filename=doc.filename,
                    type_detected="plain_text",
                    method="read",
                    pages=1,
                    quality=1.0,
                    text="texto ok",
                    logs=["fake parser finished"],
                )
            ]

        async def fake_to_thread(fn, *args, **kwargs):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

        self.main._handle_document = fake_handle_document
        self.main.asyncio.to_thread = fake_to_thread
        try:
            self.main._job_store_update(job_id, status="processing")
            self.main._append_job_log(job_id, "worker iniciou processamento")
            worker = threading.Thread(
                target=lambda: self.main._process_job_sync(job_id, req),
                daemon=True,
            )
            worker.start()

            self.assertTrue(started.wait(timeout=5), "job never started processing")

            queue_payload = self.main.queue_status()
            logs_payload = self.main.get_job_logs(job_id)
            recent_payload = self.main.get_recent_logs(limit=5)

            self.assertEqual(queue_payload["processing"], 1)
            self.assertEqual(logs_payload["status"], "processing")
            self.assertTrue(any("worker iniciou processamento" in entry for entry in logs_payload["logs"]))
            self.assertTrue(any("iniciando documento" in entry for entry in logs_payload["logs"]))
            self.assertEqual(recent_payload["items"][0]["job_id"], job_id)

            release.set()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive(), "worker thread did not finish")

            final_payload = self.main.get_job_logs(job_id)
            self.assertEqual(final_payload["status"], "done")
            self.assertTrue(any(doc["filename"] == "doc.pdf" for doc in final_payload["documents"]))
            self.assertTrue(any("job concluído" in entry for entry in final_payload["logs"]))
        finally:
            self.main._handle_document = original_handle_document
            self.main.asyncio.to_thread = original_to_thread
            release.set()


if __name__ == "__main__":
    unittest.main()
