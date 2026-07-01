import os
import cv2
import pickle
import numpy as np
from skimage.transform import resize

# ===========================
# Konstanta
# ===========================

EMPTY = 0
NOT_EMPTY = 1

MODEL_PATH = "model_baru.p"
MASK_PATH = "mask_1920_1080.png"

os.makedirs("output", exist_ok=True)

# ===========================
# Load Model
# ===========================

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

print("[INFO] Model berhasil dimuat.")

# ===========================
# Load Mask
# ===========================

mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(
        "mask_1920_1080.png tidak ditemukan!"
    )

connected_components = cv2.connectedComponentsWithStats(
    mask,
    4,
    cv2.CV_32S
)

# ===========================
# Ambil Bounding Box Slot
# ===========================

def get_parking_spots_bboxes(connected_components):

    (total_labels,
     label_ids,
     values,
     centroid) = connected_components

    slots = []

    for i in range(1, total_labels):

        x = int(values[i, cv2.CC_STAT_LEFT])
        y = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])

        slots.append((x, y, w, h))

    return slots


parking_spots = get_parking_spots_bboxes(
    connected_components
)

print(f"[INFO] Total Slot = {len(parking_spots)}")

# ===========================
# Prediksi Slot
# ===========================

def empty_or_not(spot_bgr):

    if spot_bgr is None:
        return EMPTY

    if spot_bgr.size == 0:
        return EMPTY

    try:

        img = resize(
            spot_bgr,
            (15,15,3)
        )

        flat = img.flatten().reshape(1,-1)

        pred = MODEL.predict(flat)[0]

        return pred

    except Exception:

        return EMPTY


# ===========================
# Proses 1 Frame
# ===========================

def proses_frame(frame):

    frame_copy = frame.copy()

    kosong = 0
    terisi = 0

    for (x,y,w,h) in parking_spots:

        if x<0 or y<0:
            continue

        if x+w>frame.shape[1]:
            continue

        if y+h>frame.shape[0]:
            continue

        crop = frame[
            y:y+h,
            x:x+w
        ]

        status = empty_or_not(crop)

        if status == EMPTY:

            color = (0,255,0)
            kosong += 1

        else:

            color = (0,0,255)
            terisi += 1

        cv2.rectangle(
            frame_copy,
            (x,y),
            (x+w,y+h),
            color,
            2
        )

    cv2.rectangle(
        frame_copy,
        (0,0),
        (370,50),
        (0,0,0),
        -1
    )

    cv2.putText(
        frame_copy,
        f"Kosong : {kosong}",
        (10,20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

    cv2.putText(
        frame_copy,
        f"Terisi : {terisi}",
        (10,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,0,255),
        2
    )

    return frame_copy,kosong,terisi


# ===========================
# Deteksi Video
# ===========================

def detect_video(video_path):

    print("[INFO] Memproses Video...")

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_path = "output/hasil_deteksi.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width,height)
    )

    STEP = 5

    frame_idx = 0

    last_frame = None

    kosong = 0
    terisi = 0

    while True:

        ret,frame = cap.read()

        if not ret:
            break

        if frame_idx % STEP == 0 or last_frame is None:

            hasil,kosong,terisi = proses_frame(
                frame
            )

            last_frame = hasil

        out.write(last_frame)

        frame_idx += 1

    cap.release()

    out.release()

    print("[INFO] Selesai.")

    return output_path,kosong,terisi
