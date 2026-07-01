import cv2
import pickle
import numpy as np
from skimage.transform import resize

EMPTY = 0
NOT_EMPTY = 1

with open("model_baru.p", "rb") as f:
    MODEL = pickle.load(f)

MASK_PATH = "mask_1920_1080.png"

mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

def get_parking_spots_bboxes(connected_components):
    """Ubah hasil connected components menjadi list bounding box [x, y, w, h] tiap slot."""
    (total_labels, label_ids, values, centroid) = connected_components
    slots = []
    for i in range(1, total_labels):  # index 0 = background, dilewati
        x = int(values[i, cv2.CC_STAT_LEFT])
        y = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])
        slots.append([x, y, w, h])
    return slots


mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
parking_spots = get_parking_spots_bboxes(connected_components)

print(f'Jumlah slot parkir terdeteksi dari mask: {len(parking_spots)}')


def empty_or_not(spot_bgr):
    # Cegah error jika crop kosong atau tidak valid
    if spot_bgr is None or spot_bgr.size == 0 or spot_bgr.shape[0] == 0 or spot_bgr.shape[1] == 0:
        return EMPTY  # anggap kosong untuk menghindari crash
    try:
        img_resized = resize(spot_bgr, (15, 15, 3))
    except Exception:
        return EMPTY
    flat_data = img_resized.flatten().reshape(1, -1)
    y_output = MODEL.predict(flat_data)
    return EMPTY if y_output == 0 else NOT_EMPTY

def proses_frame(frame, spots):
    annotated = frame.copy()
    slot_kosong, slot_terisi = 0, 0

    for x, y, w, h in spots:
        # Validasi slot: ukuran harus positif dan berada dalam frame
        if w <= 0 or h <= 0:
            continue
        if y + h > frame.shape[0] or x + w > frame.shape[1] or x < 0 or y < 0:
            continue
        spot_crop = frame[y:y+h, x:x+w, :]
        if spot_crop.size == 0:
            continue

        status = empty_or_not(spot_crop)

        if status == EMPTY:
            color = (0, 255, 0)   # Hijau
            slot_kosong += 1
        else:
            color = (0, 0, 255)   # Merah
            slot_terisi += 1

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

    # Tampilkan statistik
    cv2.rectangle(annotated, (0, 0), (350, 40), (0, 0, 0), -1)
    cv2.putText(annotated, f'Kosong: {slot_kosong}  Terisi: {slot_terisi}', (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated, slot_kosong, slot_terisi

    
def detect_video(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output/hasil_deteksi.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width,height)
    )

    last_kosong = 0
    last_terisi = 0

    while True:

        ret,frame = cap.read()

        if not ret:
            break

        hasil,kosong,terisi = proses_frame(
            frame,
            parking_spots
        )

        last_kosong = kosong
        last_terisi = terisi

        out.write(hasil)

    cap.release()
    out.release()

    return output_path,last_kosong,last_terisi