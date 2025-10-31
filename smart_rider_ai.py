import cv2
from ultralytics import YOLO
import os
from datetime import datetime

# ===============================
# ตั้งค่าเริ่มต้น
# ===============================
MODEL_PERSON_BIKE = "models/yolov8m.pt"   # โมเดลหลัก (คน + มอเตอร์ไซค์)
MODEL_HELMET = "models/helmet.pt"         # โมเดลตรวจหมวกกันน็อก
CAPTURE_DIR = "captures"
CONF_THRESHOLD = 0.55  # เพิ่มความมั่นใจขั้นต่ำ

# กำหนดคลาสที่ต้องการจากโมเดลหลัก (ขึ้นอยู่กับโมเดลของคุณ)
CLASS_IDS_MAIN = [0, 3]  # person=0, motorcycle=3 (ตรวจสอบได้จาก model.names)
TARGET_LABELS = ["person", "motorcycle", "helmet"]

os.makedirs(CAPTURE_DIR, exist_ok=True)

# ===============================
# โหลดโมเดล
# ===============================
print("🔄 กำลังโหลดโมเดล YOLOv8 (person+motorcycle, helmet) ...")
model_main = YOLO(MODEL_PERSON_BIKE)
model_helmet = YOLO(MODEL_HELMET)
print("✅ โหลดโมเดลสำเร็จทั้งหมด!")

# ===============================
# เปิดกล้อง
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

print("🎯 เริ่มการตรวจจับ (กด Q เพื่อออก)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ ไม่สามารถอ่านภาพจากกล้องได้")
        break

    # ===== ตรวจจับด้วยโมเดลหลัก (เฉพาะ person และ motorcycle) =====
    results_main = model_main(frame, conf=CONF_THRESHOLD, classes=CLASS_IDS_MAIN)
    detected_main = [results_main[0].names[int(box.cls[0])] for box in results_main[0].boxes]
    frame_main = results_main[0].plot()

    # ===== ตรวจจับด้วยโมเดลหมวกกันน็อก =====
    results_helmet = model_helmet(frame, conf=CONF_THRESHOLD)
    detected_helmet = [results_helmet[0].names[int(box.cls[0])] for box in results_helmet[0].boxes]
    frame_helmet = results_helmet[0].plot()

    # ===== รวมผลทั้งหมด =====
    detected_all = list(set(detected_main + detected_helmet))

    # ===== รวมภาพจากสองโมเดล =====
    annotated_frame = cv2.addWeighted(frame_main, 0.7, frame_helmet, 0.3, 0)

    # ===== เงื่อนไขบันทึกภาพ =====
    if all(item in detected_all for item in TARGET_LABELS):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{CAPTURE_DIR}/helmet_rider_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 พบคนขี่มอเตอร์ไซค์สวมหมวกกันน็อก -> {filename}")

    # ===== แสดงผล =====
    cv2.imshow("SmartRider AI - Precision Mode", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 ปิดระบบ SmartRider AI")
        break

cap.release()
cv2.destroyAllWindows()
