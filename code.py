import cv2
import numpy as np
import os
import sys
from util import empty_or_not
from sklearn.cluster import AgglomerativeClustering

# ---- Config ---- #
MASK_PATH = './mask_1920_1080.png'
VIDEO_PATH = './data/parking_1920_1080_loop.mp4'
DEBUG_DIR = "debug_frames"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ---- Command-Line Argument ---- #
valid_modes = ["kmeans", "meanshift"]
SEGMENTATION_MODE = sys.argv[1] if len(sys.argv) > 1 else "kmeans"

if SEGMENTATION_MODE not in valid_modes:
    print(f"❌ Invalid mode: {SEGMENTATION_MODE}")
    print(f"✅ Valid options: {valid_modes}")
    exit(1)


# ---- Segmentation Techniques ---- #
def greyscale_thresholding(frame_gray):
    _, mask = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def kmeans_segmentation_opencv(frame_gray, k=2):
    Z = frame_gray.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented = labels.flatten().reshape(frame_gray.shape)
    return (segmented * (255 // (k - 1))).astype(np.uint8)

def mean_shift_segmentation(frame, spatial_radius=10, color_radius=20):
    shifted = cv2.pyrMeanShiftFiltering(frame, spatial_radius, color_radius)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    return gray


def segment(frame, mode, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "kmeans":
        return kmeans_segmentation_opencv(gray, k=2)
    elif mode == "meanshift":
        return mean_shift_segmentation(frame)
    else:
        raise ValueError("Invalid SEGMENTATION_MODE selected.")


# ---- Spot Detection ---- #
def extract_masked_bounding_boxes(mask, segmented):
    masked_segment = cv2.bitwise_and(segmented, segmented, mask=mask)
    contours, _ = cv2.findContours(masked_segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]
    return boxes


# ---- Main Loop ---- #
def main():
    mask = cv2.imread(MASK_PATH, 0)
    if mask is None:
        print(f"❌ Cannot load mask image from {MASK_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video file.")
        return

    frame_nmr = 0
    step = 30
    spots_status = []

    print(f"✅ Using segmentation mode: {SEGMENTATION_MODE}")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("📦 End of video or failed frame.")
            break

        if frame_nmr % step == 0:
            segmented = segment(frame, SEGMENTATION_MODE, mask)
            boxes = extract_masked_bounding_boxes(mask, segmented)
            print(f"[Frame {frame_nmr}] → {len(boxes)} boxes detected")

            cv2.imshow("Segmented Mask (masked)", segmented)

            spots_status = []
            for (x, y, w, h) in boxes:
                crop = frame[y:y+h, x:x+w]
                status = empty_or_not(crop)
                print(f" - Spot ({x},{y}) → {'empty' if status else 'occupied'}")
                spots_status.append((x, y, w, h, status))

        for (x, y, w, h, status) in spots_status:
            color = (0, 255, 0) if status else (0, 0, 255)
            label = 'Empty' if status else 'Occupied'
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        try:
            cv2.imshow('Segmented Parking Detection', frame)
        except cv2.error:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"frame_{frame_nmr}.jpg"), frame)
            print(f"💾 Frame saved to {DEBUG_DIR}/frame_{frame_nmr}.jpg")

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("👋 Quit requested by user.")
            break

        frame_nmr += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
