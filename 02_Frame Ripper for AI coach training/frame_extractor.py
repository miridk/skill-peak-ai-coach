import os
import cv2
import glob
import sys

# ----------------------------
# CONFIG
# ----------------------------
VIDEO_PATTERN = "downloads/input.*"
OUTPUT_DIR = "frames"
INTERVAL_SECONDS = 4


def find_video():
    files = glob.glob(VIDEO_PATTERN)

    if not files:
        print("Ingen video fundet der matcher:", VIDEO_PATTERN)
        sys.exit(1)

    if len(files) > 1:
        print("Flere videoer fundet, bruger:", files[0])

    return files[0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_path = find_video()
    print("Using video:", video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * INTERVAL_SECONDS)

    frame_count = 0
    saved_count = 0

    print(f"Video FPS: {fps}")
    print(f"Saving one frame every {INTERVAL_SECONDS} seconds")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(OUTPUT_DIR, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            print("Saved:", filename)
            saved_count += 1

        frame_count += 1

    cap.release()
    print("Done.")


if __name__ == "__main__":
    main()