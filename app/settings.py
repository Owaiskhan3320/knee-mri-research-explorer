from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    queue_size: int = 3
    max_upload_bytes: int = 150 * 1024**2
    max_expanded_bytes: int = 512 * 1024**2
    max_files: int = 4000
    max_file_bytes: int = 16 * 1024**2
    max_pixels: int = 128_000_000
    timeout_seconds: int = 300
    ttl_seconds: int = 1800
    cpu_threads: int = 2
    allow_uploads: bool = True

    @property
    def runtime(self):
        return self.root / ".runtime" / "jobs"

    @property
    def model_path(self):
        return self.root / "models" / "deploy.pt"

    @classmethod
    def from_env(cls):
        return cls(
            root=Path(os.environ.get("KNEE_ROOT", str(ROOT))).resolve(),
            allow_uploads=os.environ.get("KNEE_ALLOW_UPLOADS", "1") == "1",
            timeout_seconds=int(os.environ.get("KNEE_TIMEOUT_SECONDS", "300")),
            cpu_threads=max(1, int(os.environ.get("KNEE_CPU_THREADS", "2"))),
        )
