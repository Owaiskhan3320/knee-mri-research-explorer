"""Encode a captioned walkthrough from actual generated-data browser captures.

The captures are held for readability; this is not a real-time latency recording.
The optional encoder is a local FFmpeg binary, never a hosted video service.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import textwrap
from PIL import Image


def timestamp(seconds, separator="."):
    milliseconds = round(seconds * 1000)
    return f"{milliseconds // 3600000:02}:{milliseconds // 60000 % 60:02}:{milliseconds // 1000 % 60:02}{separator}{milliseconds % 1000:03}"


def render(captures, encoder, output, font):
    frames = json.loads((captures / "frames.json").read_text())
    if not 3 <= len(frames) <= 30:
        raise ValueError("Expected 3 to 30 reviewed browser frames.")
    inputs, filters, subtitles, elapsed = ["ffconcat version 1.0"], [], ["WEBVTT\n"], 0.
    dimensions = None
    for index, frame in enumerate(frames):
        name = frame["file"]
        if not re.fullmatch(r"\d{2}-[a-z-]+\.png", name):
            raise ValueError("Use simple local capture names; no external media URLs.")
        duration = float(frame["duration"])
        if not 1 <= duration <= 15:
            raise ValueError("Frame duration must be 1 to 15 seconds.")
        with Image.open(captures / name) as screenshot:
            if dimensions is None:
                dimensions = screenshot.size
            if screenshot.size != dimensions:
                raise ValueError("All screenshots must have identical viewport dimensions.")
        caption = textwrap.fill(frame["caption"], width=max(35, int((dimensions[0] - 32) / 10)))
        caption_file = f"caption{index:02}.txt"
        (captures / caption_file).write_text(caption, encoding="utf-8")
        inputs.extend([f"file '{name}'", f"duration {duration:.3f}"])
        font_value = font.resolve().as_posix().replace(":", r"\:")
        filters.append(f"drawtext=fontfile='{font_value}':textfile='{caption_file}':fontsize=18:fontcolor=white:x=16:y=h-80:enable='gte(t,{elapsed:.3f})*lt(t,{elapsed+duration:.3f})'")
        subtitles.append(f"{timestamp(elapsed)} --> {timestamp(elapsed + duration)}\n{caption}\n")
        elapsed += duration
    inputs.append(f"file '{frames[-1]['file']}'")
    (captures / "frames.ffconcat").write_text("\n".join(inputs) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_value = "pad=ceil(iw/2)*2:ceil(ih/2)*2+96:0:0:black," + ",".join(filters)
    command = [str(encoder), "-hide_banner", "-loglevel", "warning", "-y", "-f", "concat", "-safe", "0",
               "-i", "frames.ffconcat", "-vf", filter_value, "-r", "12", "-t", str(elapsed),
               "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output.resolve())]
    result = subprocess.run(command, cwd=captures, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    output.with_suffix(".vtt").write_text("\n".join(subtitles), encoding="utf-8")
    print(json.dumps({"file": str(output), "seconds": elapsed, "capture_count": len(frames),
                      "bytes": output.stat().st_size, "source": "Actual browser captures of generated DICOMs; held frames, no narration"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/demo.mp4"))
    parser.add_argument("--font", type=Path, default=Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf")
    args = parser.parse_args()
    render(args.captures.resolve(), args.encoder.resolve(), args.output, args.font)
