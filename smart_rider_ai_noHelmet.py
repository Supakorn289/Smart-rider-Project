import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import numpy as np
import threading
import time

# ===============================
# ⚙️ ส่วนที่ 1: ตั้งค่าเริ่มต้นระบบ
# ===============================
MODEL_PERSON_BIKE = "models/yolov8n.pt"   # ✅ ใช้โมเดลเล็กลง (แนวทางที่ 1)
MODEL_HELMET = "models/helmet.pt"         # โมเดลตรวจจับหมวกกันน็อค
CAPTURE_DIR = "captures"                  # โฟลเดอร์เก็บภาพเมื่อพบการละเมิด
LOG_FILE = "logs/detection_log.txt"       # เก็บบันทึกการตรวจจับ
CONF_THRESHOLD = 0.6                      # ค่าความมั่นใจขั้นต่ำ
IOU_PERSON_BIKE = 0.25                    # ระดับการซ้อนที่ถือว่า "อยู่บนมอเตอร์ไซค์"
IOU_HELMET_HEAD = 0.15                    # ระดับการซ้อนที่ถือว่ามีหมวก

CLASS_IDS_MAIN = [0, 3]  # person=0, motorcycle=3 (YOLOv8 class IDs)

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ===============================
# 🚀 ส่วนที่ 2: โหลดโมเดล YOLO
# ===============================
print("🔄 กำลังโหลดโมเดล YOLOv8 (person + motorcycle + helmet)...")
model_main = YOLO(MODEL_PERSON_BIKE)
model_helmet = YOLO(MODEL_HELMET)
print("✅ โหลดโมเดลสำเร็จทั้งหมด!")

# ===============================
# 🧠 ฟังก์ชันคำนวณ IoU
# ===============================
def compute_iou(boxA, boxB):
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
# 🧹 ฟังก์ชันกรองผลลัพธ์ YOLO
# ===============================
def filter_detections(results, target_classes, conf_threshold):
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls in target_classes and conf >= conf_threshold:
            detections.append(box.xyxy[0].tolist())
    return detections

# ===============================
# 🎥 ส่วนที่ 3: Thread สำหรับอ่านกล้อง (แนวทางที่ 3)
# ===============================
class VideoStream:
    def __init__(self, src=0, width=640, height=480):  # ✅ ลดขนาดภาพ (แนวทางที่ 2)
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            time.sleep(0.01)  # ลดโหลด CPU

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

# ===============================
# 🎯 เริ่มต้นระบบกล้อง
# ===============================
cap = VideoStream(0)
print("🎯 เริ่มตรวจจับ (กด Q เพื่อออก)")

# ===============================
# 🔁 ส่วนหลัก: วนลูปตรวจจับ
# ===============================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ ไม่สามารถอ่านภาพจากกล้องได้")
        continue

    # ===== ตรวจจับคนและมอเตอร์ไซค์ =====
    results_main = model_main(frame, conf=CONF_THRESHOLD, classes=CLASS_IDS_MAIN)
    persons = filter_detections(results_main, [0], CONF_THRESHOLD)
    motorcycles = filter_detections(results_main, [3], CONF_THRESHOLD)
    frame_main = results_main[0].plot()

    # ===== ตรวจจับหมวกกันน็อค =====
    results_helmet = model_helmet(frame, conf=CONF_THRESHOLD)
    helmets = filter_detections(results_helmet, [0], CONF_THRESHOLD)
    frame_helmet = results_helmet[0].plot()

    # ===== รวมผลลัพธ์ของทั้งสองโมเดล =====
    annotated_frame = cv2.addWeighted(frame_main, 0.7, frame_helmet, 0.3, 0)

    # ===== ตรวจหาผู้ขับขี่ไม่สวมหมวก =====
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
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"[{timestamp}] พบผู้ไม่สวมหมวก -> {filename}\n")

    # ===== แสดงผลภาพ =====
    cv2.imshow("SmartRider AI - Helmet Detection (Smooth Mode)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 ปิดระบบ SmartRider AI")
        break

cap.release()
cv2.destroyAllWindows()
