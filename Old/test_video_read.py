import cv2

VIDEO_PATH = "Video.mp4"  # ret hvis nødvendigt

cap = cv2.VideoCapture(VIDEO_PATH)
print("isOpened:", cap.isOpened())

ok, frame = cap.read()
print("first read ok:", ok)
print("frame is None:", frame is None)
if frame is not None:
    print("frame shape:", frame.shape)

cap.release()