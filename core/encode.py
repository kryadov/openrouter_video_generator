"""H.264 encoder selection: NVIDIA NVENC when it actually works, otherwise CPU libx264.

VIDEO_ENCODER=auto (default) | nvenc | x264
"""
import functools
import subprocess

from core import config


@functools.lru_cache(maxsize=1)
def nvenc_works():
    """NVENC can be listed by ffmpeg without a usable GPU (e.g. a CPU-only VM), so try a real encode."""
    try:
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def h264_args(quality=19):
    """ffmpeg video codec args for a high-quality H.264 master."""
    want = config.get("VIDEO_ENCODER").lower()
    if want == "nvenc" or (want == "auto" and nvenc_works()):
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-cq", str(quality), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(quality - 1)]
