import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
import gradio as gr

from .constants import MODEL_ID, NOTICE
from .jobs import BusyError, JobManager
from .settings import Settings
from .ui import CSS, build_ui


class BodyLimit:
    def __init__(self, app, maximum):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid content length"}, status_code=400)(scope, receive, send)
        if declared > self.maximum:
            return await JSONResponse({"detail": "Upload exceeds the application limit"}, status_code=413)(scope, receive, send)
        total = 0
        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.maximum:
                raise HTTPException(413, "Upload exceeds the application limit")
            return message
        await self.app(scope, limited_receive, send)


def create_app(settings=None, manager=None, mount_ui=True):
    settings = settings or Settings.from_env()
    manager = manager or JobManager(settings)
    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        try:
            yield
        finally:
            manager.close()
            cache = (settings.root / ".runtime" / "gradio").resolve()
            if cache.parent == (settings.root / ".runtime").resolve() and cache.name == "gradio":
                shutil.rmtree(cache, ignore_errors=True)

    api = FastAPI(title="Knee MRI Research Explorer", version="0.1.0", lifespan=lifespan,
                  description=NOTICE)
    api.state.manager = manager
    api.add_middleware(BodyLimit, maximum=settings.max_upload_bytes + 2 * 1024**2)

    def owned_job(jid, authorization, token=None):
        credential = authorization[7:] if authorization and authorization.startswith("Bearer ") else token
        try:
            return manager.get(jid, credential)
        except KeyError:
            raise HTTPException(404, "Job not found or access expired")

    @api.get("/app", include_in_schema=False)
    def home():
        return RedirectResponse("/")

    @api.get("/health/live")
    def live():
        return {"status": "alive", "paid_services": False}

    @api.get("/health/ready")
    def ready():
        ok = manager.worker_ready.is_set()
        return JSONResponse({"ready": ok, "model_id": MODEL_ID, "device": "cpu",
                             "error": manager.startup_error}, status_code=200 if ok else 503)

    @api.get("/api/v1/examples")
    def examples():
        return [{"id": "synthetic", "name": "Generated software test", "synthetic": True,
                 "description": "Artificial DICOM phantom; not a medical example or accuracy evaluation."}]

    @api.post("/api/v1/jobs", status_code=202)
    async def submit(kind: str = Form("synthetic"), study: UploadFile | None = File(None),
                     metadata: UploadFile | None = File(None)):
        if kind not in {"synthetic", "upload"}:
            raise HTTPException(422, "kind must be synthetic or upload")
        if kind == "upload" and not settings.allow_uploads:
            raise HTTPException(403, "Uploads are disabled")
        try:
            if kind == "synthetic":
                job = manager.new_job("synthetic")
            else:
                if not study or not metadata:
                    raise ValueError("Provide a study ZIP and a metadata CSV")
                with tempfile.TemporaryDirectory(prefix="incoming_", dir=settings.runtime) as temporary:
                    paths = []
                    for upload, filename, limit in ((study, "study.zip", settings.max_upload_bytes),
                                                     (metadata, "series_metadata.csv", 1024**2)):
                        p, size = Path(temporary) / filename, 0
                        with p.open("wb") as f:
                            while block := await upload.read(1024**2):
                                size += len(block)
                                if size > limit:
                                    raise ValueError("Input file exceeds its size limit")
                                f.write(block)
                        await upload.close()
                        paths.append(p)
                    job = manager.new_job("upload", *paths)
        except BusyError as exc:
            raise HTTPException(429, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {**job.public(), "access_token": job.token,
                "status_url": f"/api/v1/jobs/{job.id}"}

    @api.get("/api/v1/jobs/{jid}")
    def status(jid: str, authorization: str | None = Header(None)):
        return owned_job(jid, authorization).public()

    @api.get("/api/v1/jobs/{jid}/result")
    def result(jid: str, authorization: str | None = Header(None), token: str | None = Query(None), download: bool = False):
        job = owned_job(jid, authorization, token)
        if job.state != "completed":
            raise HTTPException(409, "Job has no completed result")
        headers = {"Cache-Control": "no-store"}
        if download:
            headers["Content-Disposition"] = 'attachment; filename="knee-results.json"'
        return JSONResponse(job.result, headers=headers)

    @api.get("/api/v1/jobs/{jid}/previews/{index}")
    def preview(jid: str, index: int, authorization: str | None = Header(None)):
        job = owned_job(jid, authorization)
        if job.state != "completed" or not 0 <= index < 40 or index not in job.result["valid_windows"]:
            raise HTTPException(404, "Preview unavailable")
        return FileResponse(job.directory / "previews" / f"{index}.png", media_type="image/png", headers={"Cache-Control": "no-store"})

    @api.delete("/api/v1/jobs/{jid}")
    def delete(jid: str, authorization: str | None = Header(None)):
        job = owned_job(jid, authorization)
        return manager.cancel(job.id, job.token)

    if mount_ui:
        api = gr.mount_gradio_app(api, build_ui(manager), path="/", css=CSS,
                                 theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate", font=["system-ui", "sans-serif"]),
                                 max_file_size=settings.max_upload_bytes,
                                 blocked_paths=[str(settings.root / "models"), str(settings.runtime)])
    return api
