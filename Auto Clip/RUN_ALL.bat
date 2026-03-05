@echo off
cd /d "%~dp0"

echo ==========================
echo STEP 1: Detect matches
echo ==========================
python -u auto_segments.py --input input.mp4 --ref ref.jpg --out_csv segments.csv --use-motion

echo ==========================
echo STEP 2: Clip + Prepare
echo ==========================
python clip_and_prep.py --input input.mp4 --segments segments.csv --mode both --preset fast --fps 30 --scale 1920 --no-audio

echo ==========================
echo DONE
echo ==========================
pause