import sys
import subprocess
from pathlib import Path

OUT_DIR = Path("downloads")
OUT_DIR.mkdir(exist_ok=True)

def download(url):
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        str(OUT_DIR / "input.%(ext)s"),
        url
    ]

    subprocess.run(cmd)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_downloader.py <youtube_url>")
        sys.exit()

    download(sys.argv[1])