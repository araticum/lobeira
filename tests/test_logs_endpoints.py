import sys
import types
import unittest

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
    def add_task(self, *args, **kwargs):
        return None


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

import main
from fastapi import HTTPException


class LogsEndpointsTests(unittest.TestCase):
    def setUp(self):
        self.original_jobs = main.jobs.copy()
        main.jobs.clear()
        main.jobs.update(
            {
                "job-older": {
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
                },
                "job-newer": {
                    "job_id": "job-newer",
                    "tender_id": "tender-2",
                    "status": "error",
                    "created_at": 200.0,
                    "processing_time_s": 0.75,
                    "errors": ["Falha ao baixar arquivo"],
                    "logs": ["[erro] Falha ao baixar arquivo"],
                    "documents": [],
                },
            }
        )

    def tearDown(self):
        main.jobs.clear()
        main.jobs.update(self.original_jobs)

    def test_get_recent_logs_returns_newest_jobs_first_and_respects_limit(self):
        payload = main.get_recent_logs(limit=1)

        self.assertEqual(payload["count"], 1)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["job_id"], "job-newer")
        self.assertNotIn("documents", payload["items"][0])
        self.assertEqual(payload["items"][0]["logs"], ["[erro] Falha ao baixar arquivo"])

    def test_get_job_logs_returns_document_level_logs(self):
        payload = main.get_job_logs("job-older")

        self.assertEqual(payload["job_id"], "job-older")
        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["logs"], ["[doc-a.pdf] pymupdf: ok"])
        self.assertEqual(len(payload["documents"]), 1)
        self.assertEqual(payload["documents"][0]["filename"], "doc-a.pdf")
        self.assertEqual(payload["documents"][0]["logs"], ["pymupdf: ok"])

    def test_get_job_logs_returns_404_for_missing_job(self):
        with self.assertRaises(HTTPException) as ctx:
            main.get_job_logs("job-missing")

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, "Job not found")


if __name__ == "__main__":
    unittest.main()
