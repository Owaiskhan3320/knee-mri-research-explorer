"""Start the research app on loopback only; never creates a public share link."""
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import multiprocessing
from pathlib import Path
os.environ["GRADIO_TEMP_DIR"] = str(Path(__file__).resolve().parent / ".runtime" / "gradio")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import uvicorn
    from app.main import create_app
    uvicorn.run(create_app(), host=os.environ.get("KNEE_HOST", "127.0.0.1"),
                port=int(os.environ.get("KNEE_PORT", "7860")), workers=1, access_log=False)
