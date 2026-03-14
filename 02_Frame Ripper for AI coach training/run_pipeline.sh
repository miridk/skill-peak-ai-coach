#!/bin/bash

echo ""
echo "===== SKILL PEAK VIDEO PIPELINE ====="
echo ""

# -------------------------
# Ask user for video URL
# -------------------------
read -p "Paste YouTube video URL: " VIDEO_URL

if [ -z "$VIDEO_URL" ]; then
    echo "No URL provided. Exiting..."
    exit 1
fi

DOWNLOAD_DIR="./downloads"

# -------------------------
# 1. Clear downloads folder
# -------------------------
echo ""
echo "Cleaning downloads folder..."

if [ -d "$DOWNLOAD_DIR" ]; then
    rm -rf "$DOWNLOAD_DIR"/*
else
    mkdir -p "$DOWNLOAD_DIR"
fi

echo "Downloads folder ready."
echo ""

# -------------------------
# 2. Download video
# -------------------------
echo "Downloading video..."
python video_downloader.py "$VIDEO_URL"

if [ $? -ne 0 ]; then
    echo "video_downloader.py failed!"
    exit 1
fi

echo "Video download complete."
echo ""

# -------------------------
# 3. Extract frames
# -------------------------
echo "Extracting frames..."
python frame_extractor.py

if [ $? -ne 0 ]; then
    echo "frame_extractor.py failed!"
    exit 1
fi

echo ""
echo "===== PIPELINE COMPLETE ====="
echo ""