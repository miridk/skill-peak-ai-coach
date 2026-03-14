import argparse
import csv
import os
import subprocess
from pathlib import Path


def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input.mp4")
    ap.add_argument("--segments", default="segments.csv")
    ap.add_argument("--out", default="out")
    ap.add_argument("--mode", default="both")
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=1920)
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input video not found: {args.input}")

    if not os.path.exists(args.segments):
        raise SystemExit("segments.csv not found")

    ensure_dir(args.out)
    clips_dir = os.path.join(args.out, "clips")
    prep_dir = os.path.join(args.out, "prep")

    ensure_dir(clips_dir)
    ensure_dir(prep_dir)

    with open(args.segments, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=1):
            start = row["start"]
            end = row["end"]
            name = row["name"]

            print(f"\n========== Segment {i}: {name} ==========")

            clip_out = os.path.join(clips_dir, f"{name}.mp4")
            prep_out = os.path.join(prep_dir, f"{name}_prep.mp4")

            # ---------- CLIP ----------
            if args.mode in ["clip", "both"]:

                cmd_clip = [
                    "ffmpeg",
                    "-y",
                    "-ss", start,
                    "-to", end,
                    "-i", args.input,
                    "-c", "copy",
                    clip_out
                ]

                run(cmd_clip)

            # ---------- PREP ----------
            if args.mode in ["prep", "both"]:

                vf = (
                    f"scale={args.scale}:-2:flags=lanczos,"
                    f"fps={args.fps},"
                    f"hqdn3d=1.2:1.2:6:6,"
                    f"unsharp=5:5:0.9"
                )

                cmd_prep = [
                    "ffmpeg",
                    "-y",
                    "-ss", start,
                    "-to", end,
                    "-i", args.input,
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "20",
                ]

                if args.no_audio:
                    cmd_prep.append("-an")

                cmd_prep.append(prep_out)

                run(cmd_prep)

    print("\n✅ Clip + Prep finished")


if __name__ == "__main__":
    main()