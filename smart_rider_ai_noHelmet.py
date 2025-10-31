import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import numpy as np

# ===============================
# ⚙️ ส่วนที่ 1: ตั้งค่าเริ่มต้นระบบ
# ===============================
MODEL_PERSON_BIKE = "models/yolov8m.pt"   # โมเดลตรวจจับคน + มอเตอร์ไซค์
MODEL_HELMET = "models/helmet.pt"         # โมเดลตรวจจับหมวกกันน็อค
CAPTURE_DIR = "captures"                  # โฟลเดอร์เก็บภาพเมื่อพบการละเมิด
LOG_FILE = "logs/detection_log.txt"       # [🆕 เพิ่มมาใหม่] เก็บบันทึกการตรวจจับ
CONF_THRESHOLD = 0.6                      # [🆕 เพิ่ม] เพิ่มค่าความมั่นใจขั้นต่ำเพื่อกรอง noise
IOU_PERSON_BIKE = 0.25                    # [🆕 เพิ่ม] กำหนดระดับการซ้อนที่ถือว่า "อยู่บนมอเตอร์ไซค์"
IOU_HELMET_HEAD = 0.15                    # [🆕 เพิ่ม] กำหนดระดับการซ้อนที่ถือว่ามีหมวก

CLASS_IDS_MAIN = [0, 3]  # person=0, motorcycle=3 (YOLOv8 class IDs)
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)         # [🆕 เพิ่ม] สร้างโฟลเดอร์ log อัตโนมัติ

# ===============================
# 🚀 ส่วนที่ 2: โหลดโมเดล YOLO
# ===============================
print("🔄 กำลังโหลดโมเดล YOLOv8 (person + motorcycle + helmet)...")
model_main = YOLO(MODEL_PERSON_BIKE)
model_helmet = YOLO(MODEL_HELMET)
print("✅ โหลดโมเดลสำเร็จทั้งหมด!")

# ===============================
# 🧠 ส่วนที่ 3: ฟังก์ชันคำนวณ IoU (Intersection over Union)
# ===============================
def compute_iou(boxA, boxB):
    """[ปรับปรุง] คำนวณค่า IoU ระหว่างกล่อง 2 กล่อง"""
    xA, yA, xB, yB = boxA
    x1, y1, x2, y2 = boxB

    xi1, yi1 = max(xA, x1), max(yA, y1)
    xi2, yi2 = min(xB, x2), min(yB, y2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area <= 0:
        return 0.0

    boxA_area = (xB - xA) * (yB - yA)
    boxB_area = (x2 - x1) * (y2 - y1)
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ===============================
# 🧹 ส่วนที่ 4: ฟังก์ชันกรองผลลัพธ์ YOLO (เพิ่มความแม่นยำ)
# ===============================
def filter_detections(results, target_classes, conf_threshold):
    """[🆕 เพิ่ม] กรองเฉพาะคลาสและความมั่นใจที่กำหนด"""
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls in target_classes and conf >= conf_threshold:
            detections.append(box.xyxy[0].tolist())
    return detections


# ===============================
# 🎥 ส่วนที่ 5: เปิดกล้อง
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

print("🎯 เริ่มตรวจจับ (กด Q เพื่อออก)")

# ===============================
# 🔁 ส่วนที่ 6: วนลูปตรวจจับ
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ ไม่สามารถอ่านภาพจากกล้องได้")
        break

    # ===== ตรวจจับคนและมอเตอร์ไซค์ =====
    results_main = model_main(frame, conf=CONF_THRESHOLD, classes=CLASS_IDS_MAIN)
    persons = filter_detections(results_main, [0], CONF_THRESHOLD)
    motorcycles = filter_detections(results_main, [3], CONF_THRESHOLD)
    frame_main = results_main[0].plot()

    # ===== ตรวจจับหมวกกันน็อค =====
    results_helmet = model_helmet(frame, conf=CONF_THRESHOLD)
    helmets = filter_detections(results_helmet, [0], CONF_THRESHOLD)  # สมมติหมวกเป็นคลาส 0
    frame_helmet = results_helmet[0].plot()

    # ===== รวมผลภาพจาก 2 โมเดล =====
    annotated_frame = cv2.addWeighted(frame_main, 0.7, frame_helmet, 0.3, 0)

    # ===== ตรวจจับคนที่ขี่มอเตอร์ไซค์แต่ไม่มีหมวก =====
    violation_found = False
    for person_box in persons:
        has_motorcycle = any(compute_iou(person_box, moto_box) > IOU_PERSON_BIKE for moto_box in motorcycles)
        if not has_motorcycle:
            continue

        # ส่วนหัวของคน (1/3 บน)
        head_region = [
            person_box[0],
            person_box[1],
            person_box[2],
            person_box[1] + (person_box[3] - person_box[1]) / 3
        ]
        has_helmet = any(compute_iou(head_region, helmet_box) > IOU_HELMET_HEAD for helmet_box in helmets)

        if not has_helmet:
            # พบผู้ขับขี่ไม่สวมหมวกกันน็อค
            violation_found = True
            cv2.rectangle(
                annotated_frame,
                (int(person_box[0]), int(person_box[1])),
                (int(person_box[2]), int(person_box[3])),
                (0, 0, 255), 3
            )
            cv2.putText(
                annotated_frame, "No Helmet!",
                (int(person_box[0]), int(person_box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
            )

    # ===== เมื่อพบการละเมิด =====
    if violation_found:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{CAPTURE_DIR}/no_helmet_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"🚨 พบผู้ขับขี่ไม่สวมหมวกกันน็อค -> {filename}")

        # [🆕 เพิ่ม] เขียน log ลงไฟล์
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"[{timestamp}] พบผู้ไม่สวมหมวก -> {filename}\n")

    # ===== แสดงผลภาพ =====
    cv2.imshow("SmartRider AI - Helmet Detection (Enhanced)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 ปิดระบบ SmartRider AI")
        break

cap.release()
cv2.destroyAllWindows()
