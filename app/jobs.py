"""One bounded queue for UI/API, isolated worker, job capabilities and cleanup."""
from dataclasses import dataclass, field
import hmac
import multiprocessing as mp
from pathlib import Path
import queue
import secrets
import shutil
import threading
import time
import uuid

from .constants import TERMINAL
from .inputs import native_path
from .worker import worker_main


class BusyError(Exception):
    pass


@dataclass
class Job:
    id: str
    token: str
    kind: str
    directory: Path
    state: str = "queued"
    progress: float = 0.
    error: str | None = None
    result: dict | None = None
    created: float = field(default_factory=time.monotonic)
    finished: float | None = None
    cancelled: bool = False

    def public(self):
        return dict(id=self.id, state=self.state, progress=self.progress, error=self.error,
                    synthetic=self.kind == "synthetic")


class JobManager:
    def __init__(self, settings):
        self.settings = settings
        self.pending = queue.Queue(maxsize=settings.queue_size)
        self.jobs = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker_ready = threading.Event()
        self.startup_error = None
        self.process = None
        self.ctx = mp.get_context("spawn")
        self.thread = None

    def start(self):
        self.settings.runtime.mkdir(parents=True, exist_ok=True)
        # This dedicated directory is owned entirely by this app, never by an uploaded path.
        self.cleanup_stale()
        self.thread = threading.Thread(target=self._dispatch, daemon=True, name="knee-jobs")
        self.thread.start()

    def _remove_directory(self, directory):
        root, target = self.settings.runtime.resolve(), directory.resolve()
        if target.parent != root or len(target.name) != 32:
            raise RuntimeError("Refusing cleanup outside an application job directory")
        shutil.rmtree(native_path(target), ignore_errors=True)

    def cleanup_stale(self):
        for p in self.settings.runtime.iterdir():
            if p.is_dir() and len(p.name) == 32 and all(c in "0123456789abcdef" for c in p.name):
                self._remove_directory(p)

    def new_job(self, kind="upload", archive=None, metadata=None):
        with self.lock:
            if not self.worker_ready.is_set():
                raise BusyError(self.startup_error or "The model is starting. Please try again shortly.")
            self.expire()
            if self.pending.full() or len(self.jobs) >= 24:
                raise BusyError("The local queue or temporary result store is full. Clear an old result or try later.")
            jid = uuid.uuid4().hex
            directory = self.settings.runtime / jid
            directory.mkdir()
            job = Job(jid, secrets.token_urlsafe(32), kind, directory)
            try:
                if kind == "upload":
                    if not self.settings.allow_uploads:
                        raise ValueError("Uploads are disabled in this deployment.")
                    if not archive or not metadata:
                        raise ValueError("Select both the study ZIP and series metadata CSV.")
                    for source, name, limit in ((Path(archive), "study.zip", self.settings.max_upload_bytes),
                                                (Path(metadata), "series_metadata.csv", 1024**2)):
                        if not source.is_file() or source.stat().st_size > limit:
                            raise ValueError("An input file is missing or exceeds the size limit.")
                        shutil.copyfile(source, directory / name)
                self.jobs[jid] = job
                self.pending.put_nowait(jid)
            except Exception:
                self.jobs.pop(jid, None)
                self._remove_directory(directory)
                raise
            return job

    def get(self, jid, token):
        with self.lock:
            self.expire()
            job = self.jobs.get(jid)
            if not job or not isinstance(token, str) or not hmac.compare_digest(job.token, token):
                raise KeyError("Job not found or access expired.")
            return job

    def cancel(self, jid, token):
        with self.lock:
            job = self.get(jid, token)
            job.cancelled = True
            if job.state in TERMINAL or job.state == "queued":
                job.state, job.finished = "cancelled", time.monotonic()
                job.result = None
                self._remove_directory(job.directory)
                self.jobs.pop(jid, None)
            return job.public()

    def expire(self):
        now = time.monotonic()
        with self.lock:
            for jid, job in list(self.jobs.items()):
                if job.finished and now - job.finished > self.settings.ttl_seconds:
                    self._remove_directory(job.directory)
                    del self.jobs[jid]

    def _stop_worker(self):
        self.worker_ready.clear()
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
        for q in (getattr(self, "tasks", None), getattr(self, "messages", None)):
            if q is not None:
                q.cancel_join_thread()
                q.close()

    def _start_worker(self):
        self._stop_worker()
        self.tasks, self.messages = self.ctx.Queue(maxsize=1), self.ctx.Queue()
        self.process = self.ctx.Process(target=worker_main, args=(self.settings, self.tasks, self.messages), daemon=True)
        self.process.start()
        deadline = time.monotonic() + 90
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            try:
                event = self.messages.get(timeout=.25)
            except queue.Empty:
                if not self.process.is_alive():
                    break
                continue
            if event.get("event") == "ready":
                self.startup_error = None
                self.worker_ready.set()
                return True
            self.startup_error = event.get("error", "Model initialization failed.")
            return False
        self.startup_error = "The inference worker could not start. Check the model and local installation."
        return False

    def _dispatch(self):
        if not self._start_worker():
            self._stop_worker()
            return
        while not self.stop_event.is_set():
            self.expire()
            try:
                jid = self.pending.get(timeout=.25)
            except queue.Empty:
                continue
            job = self.jobs.get(jid)
            if not job or job.cancelled:
                continue
            self.tasks.put({"id": jid, "kind": job.kind, "directory": str(job.directory)})
            started, reset = time.monotonic(), False
            while not self.stop_event.is_set():
                if job.cancelled or time.monotonic() - started > self.settings.timeout_seconds or not self.process.is_alive():
                    job.state = "cancelled" if job.cancelled else "failed"
                    job.error = None if job.cancelled else "Processing deadline reached or worker exited. No prediction was generated."
                    reset = True
                    self._stop_worker()
                    break
                try:
                    message = self.messages.get(timeout=.2)
                except queue.Empty:
                    continue
                if message.get("id") != jid:
                    continue
                with self.lock:
                    for key in ("state", "progress", "error", "result"):
                        if key in message:
                            setattr(job, key, message[key])
                if job.state in TERMINAL:
                    break
            job.finished = time.monotonic()
            if job.state != "completed":
                job.result = None
                self._remove_directory(job.directory)
                if job.cancelled:
                    with self.lock:
                        self.jobs.pop(jid, None)
            if reset and not self.stop_event.is_set() and not self._start_worker():
                break

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        self._stop_worker()
        with self.lock:
            for job in self.jobs.values():
                self._remove_directory(job.directory)
