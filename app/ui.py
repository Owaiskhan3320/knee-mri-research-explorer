"""Local research interface. Both UI and API use the same JobManager."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr

from .constants import MODEL_ID, NOTICE, SLOT_NAMES, TARGETS
from .jobs import BusyError

CSS = """
.gradio-container {width:100% !important; min-width:0 !important; max-width:1240px !important; margin:auto !important; font-family:system-ui,sans-serif !important;}
.gradio-container .contain, .gradio-container .wrap {min-width:0; max-width:100%;}
#hero {padding: 26px 8px 10px;}
#hero h1 {font-size: 2.5rem; letter-spacing: -.045em; margin-bottom: 8px;}
#hero p {max-width: 820px; color: #667085; font-size: 1.03rem;}
.notice {border-left: 3px solid #0d9488; padding: 10px 16px; background: #f0fdfa; color: #134e4a; border-radius: 8px;}
.badge {display:inline-block; padding:5px 11px; margin:0 8px 7px 0; background:#e7f4f1; color:#115e59; border-radius:20px; font-size:12px; font-weight:650;}
#status {min-height:60px; padding:14px; border-radius:10px; background:#f8fafc;}
#scores {font-variant-numeric:tabular-nums;}
.download {display:inline-block; padding:10px 16px; color:white !important; background:#0f766e; border-radius:8px; text-decoration:none;}
footer {display:none !important;}
"""


def build_ui(manager):
    settings = manager.settings
    def submit(source, archive, metadata):
        kind = "synthetic" if source == "Generated software test" else "upload"
        try:
            job = manager.new_job(kind, archive, metadata)
            return {"id": job.id, "token": job.token}, "**Queued.** The local worker will process this study."
        except (ValueError, BusyError) as exc:
            raise gr.Error(str(exc))

    def cancel(current):
        if current:
            try:
                manager.cancel(current["id"], current["token"])
            except KeyError:
                pass
        return None, "**Cleared.** Any active job is being cancelled and its temporary files removed."

    def choose_source(source):
        visible = settings.allow_uploads and source == "My study (local)"
        return gr.update(visible=visible), gr.update(visible=visible)

    def refresh(current, selected_slot, selected_window):
        empty = [[name, None] for name in TARGETS]
        if not current:
            status = "**Ready · CPU inference**" if manager.worker_ready.is_set() else "**Loading model…**"
            if manager.startup_error:
                status = "**Model unavailable.** " + manager.startup_error
            return status, empty, None, "", {}
        try:
            job = manager.get(current["id"], current["token"])
        except KeyError:
            return "**Expired.** Submit the study again to create a fresh job.", empty, None, "", {}
        if job.state != "completed":
            status = f"**{job.state.title()}** · {int(job.progress * 100)}%"
            if job.error:
                status += "\n\n" + job.error
            return status, empty, None, "", {}
        result = job.result
        warnings = "\n\n".join(result["warnings"])
        status = f"**Completed locally** · {result['seconds']['total']:.1f} s · {len(result['valid_windows'])}/40 valid windows"
        if warnings:
            status += "\n\n" + warnings
        index = SLOT_NAMES.index(selected_slot) * 8 + int(selected_window) - 1
        preview = job.directory / "previews" / f"{index}.png"
        image = np.asarray(Image.open(preview).copy()) if preview.is_file() else None
        download = f'<a class="download" href="/api/v1/jobs/{job.id}/result?token={job.token}&download=true" target="_blank" rel="noopener">Download result JSON</a>'
        return status, [[k, round(result["scores"][k], 5)] for k in TARGETS], image, download, result

    with gr.Blocks(title="Knee MRI Research Explorer", analytics_enabled=False, delete_cache=(60, 1800)) as ui:
        session = gr.State(None)
        gr.HTML('''<div id="hero"><span class="badge">LOCAL CPU</span><span class="badge">CONVNEXT-TINY · 12 FINDINGS</span><h1>Knee MRI Research Explorer</h1><p>Inspect the model's sampled MRI inputs and explore reproducible study-level predictions.</p></div><div class="notice">Research demonstration only. Not for diagnosis or patient care.</div>''')
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("### 1 · Choose an input")
                choices = ["Generated software test"] + (["My study (local)"] if settings.allow_uploads else [])
                source = gr.Radio(choices, value=choices[0], label="Input source")
                gr.Markdown("The generated example is an artificial intensity phantom, **not a knee scan**. It tests the software; its model scores have no medical meaning.")
                archive = gr.File(label="Study ZIP · local upload", file_types=[".zip"], type="filepath", visible=False, height=110)
                metadata = gr.File(label="Series metadata CSV", file_types=[".csv"], type="filepath", visible=False, height=110)
                with gr.Row():
                    run = gr.Button("Run local analysis", variant="primary")
                    clear = gr.Button("Clear study")
                status = gr.Markdown("**Loading model…**", elem_id="status")
            with gr.Column(scale=6):
                gr.Markdown("### 2 · Inspect model inputs")
                viewer = gr.Image(label="Centre slice of a sampled 3-slice model window", type="numpy", interactive=False,
                                  image_mode="L", format="png", height=350, buttons=["fullscreen"])
                slot = gr.Dropdown(SLOT_NAMES, value=SLOT_NAMES[0], label="Input slot")
                window = gr.Slider(1, 8, value=4, step=1, label="Sampled window")
                gr.Markdown("Displays the actual cropped model input. An empty view means that window was unavailable. This is not a lesion-localization viewer.")
            with gr.Column(scale=4):
                gr.Markdown("### 3 · Model scores")
                scores = gr.Dataframe(headers=["Finding", "Score"], value=[[x, None] for x in TARGETS],
                                      datatype=["str", "number"], interactive=False, elem_id="scores")
                gr.Markdown("Scores are uncalibrated sigmoid outputs, not diagnostic confidence or a positive/negative decision.")
                download = gr.HTML()
        with gr.Accordion("Model provenance and result details", open=False):
            gr.Markdown(f"**Model:** `{MODEL_ID}` · project-trained ConvNeXt-Tiny MIL, selected epoch 8. Confirmed competition public macro ROC AUC: **0.833**. This app does not use the public S6 model that scored 0.950.\n\nTraining used public report-derived weak labels. The small expert set was reused for development; this is not clinical validation.\n\nInputs and results remain on the machine running this app. Jobs expire 30 minutes after completion; interface file caches expire after approximately 30 minutes. Temporary files are cleared on shutdown. No training, paid API or cloud inference is performed.")
            details = gr.JSON(label="Inference result and provenance")
        gr.Markdown("Built with PyTorch · pydicom · FastAPI · Gradio. Local research software.")
        run.click(submit, [source, archive, metadata], [session, status], api_name=False)
        source.change(choose_source, [source], [archive, metadata], api_name=False, queue=False)
        clear.click(cancel, [session], [session, status], api_name=False)
        timer = gr.Timer(1)
        timer.tick(refresh, [session, slot, window], [status, scores, viewer, download, details], api_name=False, queue=False)
    return ui
