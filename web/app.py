from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, send_from_directory
import cv2
import sqlite3
import os
from datetime import datetime, timedelta
import threading
import time
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'smart_rider_secret_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CAPTURE_FOLDER'] = 'captures'
app.config['MANUAL_UPLOAD_FOLDER'] = 'manual_uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# อนุญาตนามสกุลไฟล์
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}

# ===============================
# ⚙️ ตั้งค่าโมเดล AI (ตามโค้ดเดิม)
# ===============================
MODEL_PERSON_BIKE = "models/yolov8n.pt"
MODEL_HELMET = "models/helmet.pt"
CAPTURE_DIR = "captures"
LOG_FILE = "logs/detection_log.txt"
CONF_THRESHOLD = 0.6
IOU_PERSON_BIKE = 0.25
IOU_HELMET_HEAD = 0.15
CLASS_IDS_MAIN = [0, 3]  # person=0, motorcycle=3

# สร้างโฟลเดอร์ที่จำเป็น
folders = ['database', 'uploads', 'captures', 'manual_uploads', 'logs', 'models', 'temp']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ===============================
# 🧠 AI Detection System (ตามโค้ดเดิม)
# ===============================
class AIDetectionSystem:
    def __init__(self):
        self.model_main = None
        self.model_helmet = None
        self.load_models()
        logger.info("✅ โหลดโมเดล AI สำเร็จ!")
    
    def load_models(self):
        """โหลดโมเดล YOLO"""
        try:
            logger.info("🔄 กำลังโหลดโมเดล YOLOv8 (person + motorcycle + helmet)...")
            self.model_main = YOLO(MODEL_PERSON_BIKE)
            self.model_helmet = YOLO(MODEL_HELMET)
            logger.info("✅ โหลดโมเดลสำเร็จทั้งหมด!")
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    
    def compute_iou(self, boxA, boxB):
        """คำนวณ Intersection over Union"""
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
    
    def filter_detections(self, results, target_classes, conf_threshold):
        """กรองผลลัพธ์การตรวจจับ"""
        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in target_classes and conf >= conf_threshold:
                    detections.append(box.xyxy[0].tolist())
        return detections
    
    def detect_helmets(self, frame):
        """ตรวจจับผู้ไม่สวมหมวกกันน็อค (ตามโค้ดเดิม)"""
        if self.model_main is None or self.model_helmet is None:
            return frame, False, []
        
        try:
            # ===== ตรวจจับคนและมอเตอร์ไซค์ =====
            results_main = self.model_main(frame, conf=CONF_THRESHOLD, classes=CLASS_IDS_MAIN)
            persons = self.filter_detections(results_main, [0], CONF_THRESHOLD)
            motorcycles = self.filter_detections(results_main, [3], CONF_THRESHOLD)
            frame_main = results_main[0].plot()

            # ===== ตรวจจับหมวกกันน็อค =====
            results_helmet = self.model_helmet(frame, conf=CONF_THRESHOLD)
            helmets = self.filter_detections(results_helmet, [0], CONF_THRESHOLD)
            frame_helmet = results_helmet[0].plot()

            # ===== รวมผลลัพธ์ของทั้งสองโมเดล =====
            annotated_frame = cv2.addWeighted(frame_main, 0.7, frame_helmet, 0.3, 0)

            # ===== ตรวจหาผู้ขับขี่ไม่สวมหมวก =====
            violation_found = False
            violations = []
            
            for person_box in persons:
                has_motorcycle = any(self.compute_iou(person_box, moto_box) > IOU_PERSON_BIKE for moto_box in motorcycles)
                if not has_motorcycle:
                    continue

                # ส่วนหัวของคน (1/3 บน)
                head_region = [
                    person_box[0],
                    person_box[1],
                    person_box[2],
                    person_box[1] + (person_box[3] - person_box[1]) / 3
                ]
                has_helmet = any(self.compute_iou(head_region, helmet_box) > IOU_HELMET_HEAD for helmet_box in helmets)

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
                    
                    violations.append({
                        'bbox': person_box,
                        'timestamp': datetime.now()
                    })

            return annotated_frame, violation_found, violations
            
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดในการตรวจจับ: {e}")
            return frame, False, []

# ===============================
# 🎥 Video Stream with Real AI Detection
# ===============================
INFERENCE_FPS = 5  # ปรับได้ (แนะนำ 4-8) เพื่อความลื่น

class VideoStream:
    """Decoupled capture + inference for smoother display."""
    def __init__(self, src=0, width=640, height=480, inference_fps=INFERENCE_FPS):
        self.cap = None
        self.camera_index = src
        self.width = width
        self.height = height

        # frames
        self.raw_frame = None        # latest captured frame (BGR)
        self.display_frame = None    # latest annotated/display frame (BGR)
        self.lock = threading.Lock()

        # control
        self.running = True
        self.fallback_mode = False

        # stats and AI
        self.ai_system = AIDetectionSystem()
        self.detection_stats = {'total_frames': 0, 'violations_detected': 0, 'last_violation': None}
        self.inference_interval = 1.0 / max(1, inference_fps)

        # start camera
        self.cap = self.initialize_camera()
        if self.cap is None:
            logger.warning("❌ ไม่สามารถเปิดกล้องได้ — ใช้ demo fallback")
            self.fallback_mode = True

        # threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.capture_thread.start()
        self.infer_thread.start()
        logger.info("🎥 VideoStream threads started (capture + inference)")
    
    def initialize_camera(self):
        """เริ่มต้นกล้อง (เหมือนเดิม)"""
        try:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(self.camera_index, backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"✅ เปิดกล้องสำเร็จด้วย backend: {backend}")
                            return cap
                        else:
                            cap.release()
                except Exception:
                    continue
            return None
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดในการเริ่มต้นกล้อง: {e}")
            return None
    
    def _capture_loop(self):
        """อ่านเฟรมจากกล้องให้เร็วที่สุด (ไม่ทำ inference)"""
        while self.running:
            if self.fallback_mode:
                # ให้ demo frame ในกรณี fallback
                with self.lock:
                    self.raw_frame = self.create_ai_demo_frame()
                time.sleep(0.05)
                continue

            if not self.cap or not self.cap.isOpened():
                time.sleep(0.2)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.debug("⚠️ capture: failed to read frame")
                time.sleep(0.05)
                continue

            # store latest raw frame
            with self.lock:
                self.raw_frame = frame
            # tiny sleep to avoid tight loop starving CPU
            time.sleep(0.005)
    
    def _inference_loop(self):
        """รัน AI เป็นช่วง ๆ ตาม inference_interval"""
        while self.running:
            start = time.time()
            frame_to_process = None
            with self.lock:
                if self.raw_frame is not None:
                    frame_to_process = self.raw_frame.copy()

            if frame_to_process is None:
                # ไม่มีเฟรม ให้รอแล้วลองอีกครั้ง
                time.sleep(0.05)
                continue

            try:
                # run detection (heavy) -- at limited fps
                processed_frame, violation_detected, violations = self.ai_system.detect_helmets(frame_to_process)

                # update display frame and stats
                with self.lock:
                    self.display_frame = processed_frame
                    self.detection_stats['total_frames'] += 1
                    if violation_detected:
                        self.detection_stats['violations_detected'] += 1
                        self.detection_stats['last_violation'] = datetime.now().isoformat()

                # save violation (non-blocking by delegating to thread may be done; keep simple)
                if violation_detected:
                    try:
                        self.save_violation(frame_to_process, violations)
                    except Exception as e:
                        logger.error(f"❌ ข้อผิดพลาดในการบันทึก violation: {e}")

            except Exception as e:
                logger.error(f"❌ ข้อผิดพลาดใน inference: {e}")

            # sleep to honor target inference rate
            elapsed = time.time() - start
            to_sleep = self.inference_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
    
    def run_fallback_mode(self):
        """ยังคงมีเพื่อความเข้ากันได้ แต่ไม่ได้ใช้หลักการแล้ว"""
        while self.running:
            with self.lock:
                self.display_frame = self.create_ai_demo_frame()
            time.sleep(0.1)
    
    def create_ai_demo_frame(self):
        """สร้างเฟรมทดแทนที่แสดงข้อมูล AI"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (30, 30, 50)  # สีน้ำเงินเข้ม
        
        # หัวข้อระบบ
        cv2.putText(frame, "🤖 SMART RIDER AI SYSTEM", (120, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # สถานะ
        cv2.putText(frame, "🔴 CAMERA OFFLINE - AI DEMO MODE", (80, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        # ข้อมูลการตรวจจับ
        cv2.putText(frame, f"📊 Frames Processed: {self.detection_stats['total_frames']}", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"🚨 Violations Detected: {self.detection_stats['violations_detected']}", (50, 170), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # วาดกรอบจำลองการตรวจจับ
        cv2.rectangle(frame, (100, 200), (300, 400), (0, 255, 0), 2)
        cv2.putText(frame, "AI DETECTION AREA", (110, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # วาดวัตถุจำลอง
        current_time = time.time()
        bike_x = 150 + int(100 * np.sin(current_time))
        person_x = 200 + int(80 * np.cos(current_time * 0.8))
        
        # วาดมอเตอร์ไซค์
        cv2.rectangle(frame, (bike_x, 300), (bike_x + 60, 350), (255, 100, 100), -1)
        cv2.putText(frame, "BIKE", (bike_x + 10, 325), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # วาดคน
        cv2.rectangle(frame, (person_x, 250), (person_x + 30, 300), (100, 100, 255), -1)
        cv2.putText(frame, "PERSON", (person_x - 10, 245), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # แสดงเวลา
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"🕒 {timestamp}", (250, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def save_violation(self, original_frame, violations):
        """บันทึกเหตุการณ์ละเมิด"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"no_helmet_{timestamp}.jpg"
            filepath = os.path.join(app.config['CAPTURE_FOLDER'], filename)
            
            # บันทึกรูป
            cv2.imwrite(filepath, original_frame)
            
            # บันทึกข้อมูลใน database
            conn = sqlite3.connect('database/smart_rider.db')
            c = conn.cursor()
            c.execute('''INSERT INTO events 
                        (type, description, image_path, location, timestamp) 
                        VALUES (?, ?, ?, ?, ?)''',
                     ('no_helmet', 
                      f'ตรวจพบผู้ขับขี่ไม่สวมหมวกกันน็อค - จำนวน {len(violations)} คน',
                      filename,
                      'จุดตรวจสอบหลัก',
                      datetime.now()))
            conn.commit()
            conn.close()
            
            # บันทึก log
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] พบผู้ไม่สวมหมวก {len(violations)} คน -> {filename}\n")
            
            logger.info(f"🚨 บันทึกเหตุการณ์: {filename}")
            
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดในการบันทึกเหตุการณ์: {e}")
    
    def get_frame(self):
        """รับเฟรมปัจจุบัน (เอา display_frame ก่อน ถ้าไม่มีใช้ raw_frame)"""
        try:
            with self.lock:
                frame = self.display_frame if self.display_frame is not None else self.raw_frame
                if frame is None:
                    return None
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    return jpeg.tobytes()
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดใน get_frame: {e}")
        return None
    
    def get_detection_stats(self):
        """รับสำเนาสถิติ"""
        return self.detection_stats.copy()
    
    def stop(self):
        """หยุด threads และปล่อยกล้อง"""
        self.running = False
        try:
            if self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.infer_thread.is_alive():
                self.infer_thread.join(timeout=1.0)
        except Exception:
            pass
        if self.cap and self.cap.isOpened():
            try:
                self.cap.release()
            except:
                pass
        cv2.destroyAllWindows()
        logger.info("🛑 VideoStream หยุดแล้ว (threads stopped)")

# ===============================
# 📊 Systems Manager
# ===============================
class FileManager:
    def __init__(self):
        self.max_files_per_folder = 1000
        self.max_storage_days = 30
    
    def cleanup_old_files(self):
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_storage_days)
            folders_to_clean = [app.config['CAPTURE_FOLDER'], app.config['MANUAL_UPLOAD_FOLDER']]
            
            for folder in folders_to_clean:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        filepath = os.path.join(folder, filename)
                        if os.path.isfile(filepath):
                            file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                            if file_time < cutoff_date:
                                try:
                                    os.remove(filepath)
                                    logger.info(f"🧹 ลบไฟล์เก่า: {filename}")
                                except:
                                    pass
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดในการทำความสะอาด: {e}")
    
    def get_folder_size(self, folder):
        total_size = 0
        if os.path.exists(folder):
            for dirpath, dirnames, filenames in os.walk(folder):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
        return total_size

class StatisticsManager:
    def get_daily_stats(self, days=7):
        try:
            conn = sqlite3.connect('database/smart_rider.db')
            c = conn.cursor()
            
            c.execute('''
                SELECT DATE(timestamp), type, COUNT(*) 
                FROM events 
                WHERE timestamp >= date('now', ?) 
                GROUP BY DATE(timestamp), type 
                ORDER BY DATE(timestamp)
            ''', (f'-{days} days',))
            
            stats = c.fetchall()
            conn.close()
            
            dates = []
            no_helmet_data = []
            
            current_date = datetime.now().date()
            date_list = [(current_date - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days-1, -1, -1)]
            
            for date in date_list:
                dates.append(date)
                no_helmet_count = sum(1 for s in stats if s[0] == date and s[1] == 'no_helmet')
                no_helmet_data.append(no_helmet_count)
            
            return {
                'dates': dates,
                'no_helmet': no_helmet_data
            }
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดในการโหลดสถิติ: {e}")
            return {'dates': [], 'no_helmet': []}

# ===============================
# 🏁 เริ่มต้นระบบ
file_manager = FileManager()
stats_manager = StatisticsManager()
video_stream = None

def generate_frames():
    """สร้าง video feed สำหรับ streaming"""
    while True:
        try:
            if video_stream is None:
                # ส่งภาพสถานะแทนเพื่อไม่ให้หน้าเว็บแสดงจอดำ
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                error_frame[:] = (40, 40, 60)
                cv2.putText(error_frame, "CAMERA OFFLINE / STARTING...", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                ret, jpeg = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue

            frame_bytes = video_stream.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดใน generate_frames: {e}")
            time.sleep(0.1)

# ===============================
# 🗄️ Database
# ===============================
def init_db():
    try:
        conn = sqlite3.connect('database/smart_rider.db')
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      role TEXT DEFAULT 'user')''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS events
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      type TEXT NOT NULL,
                      description TEXT,
                      image_path TEXT,
                      location TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        
        try:
            c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)", 
                     ('admin', 'admin123', 'admin'))
        except:
            pass
        
        conn.commit()
        conn.close()
        logger.info("✅ ฐานข้อมูลเริ่มต้นสำเร็จ")
    except Exception as e:
        logger.error(f"❌ ข้อผิดพลาดฐานข้อมูล: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_file_size(size_bytes):
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} bytes"

# ===============================
# 🌐 Routes
# ===============================
_app_initialized = False

@app.before_request
def initialize_system():
    global _app_initialized, video_stream
    if not _app_initialized:
        logger.info("🚀 เริ่มต้นระบบ Smart Rider AI...")
        init_db()
        file_manager.cleanup_old_files()

        # เริ่ม VideoStream เฉพาะใน process หลักของ reloader (ป้องกันเปิดกล้องซ้ำ)
        try:
            should_start_camera = (os.environ.get('WERKZEUG_RUN_MAIN') == 'true') or (not app.debug)
            if video_stream is None and should_start_camera:
                try:
                    video_stream = VideoStream()
                    logger.info("✅ VideoStream ถูกเริ่มแล้ว")
                except Exception as e:
                    logger.error(f"❌ ไม่สามารถเริ่ม VideoStream: {e}")
                    video_stream = None
            else:
                logger.info("ℹ️ ไม่เริ่ม VideoStream ใน process นี้ (reloader parent or debug skip)")
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดขณะพยายามเริ่มระบบ: {e}")

        _app_initialized = True
        logger.info("✅ ระบบพร้อมทำงาน!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            conn = sqlite3.connect('database/smart_rider.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                     (username, password))
            user = c.fetchone()
            conn.close()
            
            if user:
                session['user_id'] = user[0]
                session['username'] = user[1]
                session['role'] = user[3]
                logger.info(f"✅ ผู้ใช้ {username} เข้าสู่ระบบ")
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error='ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง')
        except:
            return render_template('login.html', error='เกิดข้อผิดพลาดในการเข้าสู่ระบบ')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        conn = sqlite3.connect('database/smart_rider.db')
        c = conn.cursor()
        
        # สถิติพื้นฐาน
        c.execute("SELECT COUNT(*) FROM events")
        total_events = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM events WHERE type = 'no_helmet'")
        no_helmet_count = c.fetchone()[0]
        
        # เหตุการณ์ล่าสุด
        c.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT 5")
        recent_events = c.fetchall()
        
        # สถิติการตรวจจับแบบ real-time (guard เมื่อ video_stream ยังไม่เริ่ม)
        detection_stats = video_stream.get_detection_stats() if video_stream else {
            'total_frames': 0, 'violations_detected': 0, 'last_violation': None
        }
        
        capture_size = file_manager.get_folder_size(app.config['CAPTURE_FOLDER'])
        
        conn.close()
        
        daily_stats = stats_manager.get_daily_stats(7)
        
        return render_template('dashboard.html',
                             username=session['username'],
                             total_events=total_events,
                             no_helmet_count=no_helmet_count,
                             recent_events=recent_events,
                             daily_stats=daily_stats,
                             detection_stats=detection_stats,
                             capture_size=format_file_size(capture_size))
    except Exception as e:
        logger.error(f"❌ ข้อผิดพลาดใน dashboard: {e}")
        return render_template('error.html', error="เกิดข้อผิดพลาดในการโหลดแดชบอร์ด")

@app.route('/live')
def live():
    """หน้าแสดงกล้องสดพร้อม AI Detection"""
    detection_stats = video_stream.get_detection_stats() if video_stream else {
        'total_frames': 0, 'violations_detected': 0, 'last_violation': None
    }
    return render_template('live.html', detection_stats=detection_stats)

# Provide the video MJPEG stream endpoint used by the live page
@app.route('/video_feed')
def video_feed():
    """Video feed ที่มีการตรวจจับ AI แบบ real-time"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detection_stats')
def api_detection_stats():
    """API สำหรับรับสถิติการตรวจจับแบบ real-time"""
    stats = video_stream.get_detection_stats() if video_stream else {
        'total_frames': 0, 'violations_detected': 0, 'last_violation': None
    }
    return jsonify(stats)

@app.route('/test_ai')
def test_ai():
    """หน้าทดสอบ AI Detection"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🧪 Smart Rider - AI Test</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .video-container {
                text-align: center;
                margin: 20px 0;
                padding: 20px;
                background: rgba(0,0,0,0.3);
                border-radius: 15px;
            }
            img {
                max-width: 90%;
                border: 3px solid #00ff00;
                border-radius: 10px;
                box-shadow: 0 0 30px rgba(0,255,0,0.5);
            }
            .stats-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .stat-card {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .stat-value {
                font-size: 2em;
                font-weight: bold;
                color: #00ff00;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 SMART RIDER AI DETECTION TEST</h1>
                <p>Real-time Helmet Detection using YOLOv8</p>
            </div>
            
            <div class="stats-container" id="statsContainer">
                <div class="stat-card">
                    <h3>📊 Total Frames</h3>
                    <div class="stat-value" id="totalFrames">0</div>
                </div>
                <div class="stat-card">
                    <h3>🚨 Violations Detected</h3>
                    <div class="stat-value" id="violations">0</div>
                </div>
                <div class="stat-card">
                    <h3>🕒 Last Detection</h3>
                    <div class="stat-value" id="lastDetection">-</div>
                </div>
            </div>
            
            <div class="video-container">
                <h2>🎥 REAL-TIME AI DETECTION</h2>
                <img src="/video_feed" alt="AI Video Feed">
                <p><em>Live AI Processing: Person + Motorcycle + Helmet Detection</em></p>
            </div>
            
            <div style="text-align: center; margin-top: 20px;">
                <a href="/" style="color: white; text-decoration: none; background: #4CAF50; padding: 10px 20px; border-radius: 5px;">
                    🏠 Back to Home
                </a>
            </div>
        </div>
        
        <script>
            function updateStats() {
                fetch('/api/detection_stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('totalFrames').textContent = data.total_frames;
                        document.getElementById('violations').textContent = data.violations_detected;
                        
                        if (data.last_violation) {
                            const date = new Date(data.last_violation);
                            document.getElementById('lastDetection').textContent = 
                                date.toLocaleTimeString();
                        }
                    });
            }
            
            // อัพเดททุก 2 วินาที
            setInterval(updateStats, 2000);
            updateStats();
        </script>
    </body>
    </html>
    '''

# Routes อื่นๆ (events, upload, manage_files) เหมือนเดิม
@app.route('/events')
def events():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    event_type = request.args.get('type', 'all')
    
    conn = sqlite3.connect('database/smart_rider.db')
    c = conn.cursor()
    
    if event_type == 'all':
        c.execute("SELECT * FROM events ORDER BY timestamp DESC")
    else:
        c.execute("SELECT * FROM events WHERE type = ? ORDER BY timestamp DESC", (event_type,))
    
    events = c.fetchall()
    conn.close()
    
    return render_template('events.html', events=events, event_type=event_type)

@app.route('/upload')
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/manage_files')
def manage_files():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    ai_files = []
    if os.path.exists(app.config['CAPTURE_FOLDER']):
        for filename in os.listdir(app.config['CAPTURE_FOLDER']):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(app.config['CAPTURE_FOLDER'], filename)
                ai_files.append({
                    'name': filename,
                    'size': format_file_size(os.path.getsize(filepath)),
                    'date': datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                })
    
    return render_template('manage_files.html', ai_files=ai_files)

if __name__ == '__main__':
    print("=" * 70)
    print("🤖 SMART RIDER AI SYSTEM")
    print("🎯 Real-time Helmet Detection with YOLOv8")
    print("📧 Login: admin / admin123")
    print("🌐 Main URL: http://localhost:5000")
    print("🧪 AI Test: http://localhost:5000/test_ai")
    print("📺 Live AI: http://localhost:5000/live")
    print("=" * 70)
    try:
        # รันปกติ (startup ของ camera จะทำใน before_request สำหรับ process ที่เหมาะสม)
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 หยุดระบบโดยผู้ใช้")
    except Exception as e:
        logger.error(f"❌ ไม่สามารถเริ่มเซิร์ฟเวอร์ได้: {e}")
    finally:
        if video_stream:
            try:
                video_stream.stop()
            except Exception:
                pass
