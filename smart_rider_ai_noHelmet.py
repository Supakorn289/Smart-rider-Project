import cv2
from ultralytics import YOLO
import os
from datetime import datetime

# ===============================
# ⚙️ ตั้งค่าเริ่มต้น
# ===============================
MODEL_PERSON_BIKE = "models/yolov8m.pt"   # ตรวจคน + มอเตอร์ไซค์
MODEL_HELMET = "models/helmet.pt"         # ตรวจหมวกกันน็อค
CAPTURE_DIR = "captures"
CONF_THRESHOLD = 0.55

CLASS_IDS_MAIN = [0, 3]  # person=0, motorcycle=3
os.makedirs(CAPTURE_DIR, exist_ok=True)

# ===============================
# 🚀 โหลดโมเดล
# ===============================
print("🔄 กำลังโหลดโมเดล YOLOv8 (person + motorcycle + helmet)...")
model_main = YOLO(MODEL_PERSON_BIKE)
model_helmet = YOLO(MODEL_HELMET)
print("✅ โหลดโมเดลสำเร็จทั้งหมด!")

# ===============================
# 🧠 ฟังก์ชันตรวจตำแหน่งซ้อนทับ (IoU)
# ===============================
def is_overlapping(boxA, boxB, threshold=0.2):
    """เช็คว่ากรอบสองกรอบซ้อนทับกันมากกว่า threshold หรือไม่"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou > threshold

# ===============================
# 🎥 เปิดกล้อง
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

print("🎯 เริ่มตรวจจับ (กด Q เพื่อออก)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ ไม่สามารถอ่านภาพจากกล้องได้")
        break

    # ===== ตรวจจับคนและมอเตอร์ไซค์ =====
    results_main = model_main(frame, conf=CONF_THRESHOLD, classes=CLASS_IDS_MAIN)
    frame_main = results_main[0].plot()

    persons = []
    motorcycles = []

    for box in results_main[0].boxes:
        cls = int(box.cls[0])
        xyxy = box.xyxy[0].tolist()
        if results_main[0].names[cls] == "person":
            persons.append(xyxy)
        elif results_main[0].names[cls] == "motorcycle":
            motorcycles.append(xyxy)

    # ===== ตรวจจับหมวกกันน็อค =====
    results_helmet = model_helmet(frame, conf=CONF_THRESHOLD)
    helmets = [box.xyxy[0].tolist() for box in results_helmet[0].boxes]
    frame_helmet = results_helmet[0].plot()

    # ===== รวมผลการตรวจจับ (Overlay 2 โมเดล) =====
    annotated_frame = cv2.addWeighted(frame_main, 0.7, frame_helmet, 0.3, 0)

    # ===== ตรวจจับ “คนขี่มอเตอร์ไซค์ที่ไม่มีหมวกกันน็อค” =====
    violation_found = False
    for person_box in persons:
        # หามอเตอร์ไซค์ที่ซ้อนกับคน
        has_motorcycle = any(is_overlapping(person_box, moto_box, 0.2) for moto_box in motorcycles)
        if not has_motorcycle:
            continue

        # หาหมวกที่อยู่บนหัว (บริเวณ 1/3 บนของกล่องคน)
        head_region = [
            person_box[0],
            person_box[1],
            person_box[2],
            person_box[1] + (person_box[3] - person_box[1]) / 3
        ]
        has_helmet = any(is_overlapping(head_region, helmet_box, 0.1) for helmet_box in helmets)

        if not has_helmet:
            # พบผู้ขับขี่ไม่สวมหมวกกันน็อค
            violation_found = True
            cv2.rectangle(
                annotated_frame,
                (int(person_box[0]), int(person_box[1])),
                (int(person_box[2]), int(person_box[3])),
                (0, 0, 255),
                3
            )
            cv2.putText(
                annotated_frame, "No Helmet!",
                (int(person_box[0]), int(person_box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
            )

    # ===== บันทึกภาพเมื่อพบการละเมิด =====
    if violation_found:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{CAPTURE_DIR}/no_helmet_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"🚨 พบผู้ขับขี่ไม่สวมหมวกกันน็อค -> {filename}")

    # ===== แสดงภาพ =====
    cv2.imshow("SmartRider AI - Spatial No Helmet Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 ปิดระบบ SmartRider AI")
        break

cap.release()
cv2.destroyAllWindows()
