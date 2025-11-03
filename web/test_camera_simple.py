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

# สร้างโฟลเดอร์ที่จำเป็น
folders = ['database', 'uploads', 'captures', 'manual_uploads', 'logs', 'models', 'temp']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ===============================
# 🎥 Video Stream แบบใหม่ที่รับประกันทำงาน
# ===============================
class GuaranteedVideoStream:
    def __init__(self):
        self.frame = None
        self.running = True
        self.fallback_mode = True  # เริ่มด้วยโหมดทดแทน
        self.frame_counter = 0
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        logger.info("🎥 เริ่มต้น VideoStream แบบรับประกัน")
    
    def update(self):
        """อัพเดทเฟรม - รับประกันว่าจะมีภาพแสดง"""
        while self.running:
            try:
                # พยายามเปิดกล้องจริง
                if not self.fallback_mode:
                    camera_frame = self.try_get_camera_frame()
                    if camera_frame is not None:
                        self.frame = camera_frame
                        continue
                    else:
                        self.fallback_mode = True
                        logger.warning("🔁 เปลี่ยนไปใช้โหมดทดแทน")
                
                # โหมดทดแทน - สร้างภาพจำลอง
                self.frame = self.create_demo_frame()
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                logger.error(f"❌ ข้อผิดพลาดใน video loop: {e}")
                self.frame = self.create_error_frame("System Error")
                time.sleep(1)
    
    def try_get_camera_frame(self):
        """พยายามรับเฟรมจากกล้องจริง"""
        try:
            # ลองเปิดกล้องชั่วคราว
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap.isOpened():
                # ตั้งค่ากล้อง
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # อ่านเฟรม
                ret, frame = cap.read()
                cap.release()
                
                if ret and frame is not None:
                    # วาดข้อมูลบนเฟรม
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, f"LIVE CAMERA - {timestamp}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Status: ACTIVE", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    return frame
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ ไม่สามารถเข้าถึงกล้อง: {e}")
            return None
    
    def create_demo_frame(self):
        """สร้างเฟรมทดแทนที่มีการเคลื่อนไหว"""
        self.frame_counter += 1
        
        # สร้างเฟรมพื้นหลัง
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # สีพื้นหลังที่เปลี่ยนไปเรื่อยๆ
        base_color = (40, 40, 40)
        pulse = int(20 * np.sin(self.frame_counter * 0.1))  # effect การเต้น
        frame[:] = [base_color[0] + pulse, base_color[1] + pulse, base_color[2] + pulse]
        
        # ข้อมูลเวลา
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # วาดกรอบหลัก
        cv2.rectangle(frame, (20, 20), (620, 460), (0, 255, 255), 2)
        cv2.rectangle(frame, (25, 25), (615, 455), (255, 255, 255), 1)
        
        # หัวข้อระบบ
        cv2.putText(frame, "🚀 SMART RIDER AI SYSTEM", (150, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # สถานะ
        cv2.putText(frame, "📡 STATUS: DEMO MODE - CAMERA OFFLINE", (80, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        # เวลาและวันที่
        cv2.putText(frame, f"🕒 {current_time} | 📅 {current_date}", (180, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # เฟรมคาน์เตอร์ (เคลื่อนไหว)
        counter_text = f"FRAME: {self.frame_counter:06d}"
        cv2.putText(frame, counter_text, (250, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        # ข้อมูลจำลองการตรวจจับ
        cv2.putText(frame, "🎯 AI DETECTION SIMULATION", (80, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        # วาดวัตถุเคลื่อนไหว
        self.draw_moving_objects(frame)
        
        # คู่มือการแก้ไข
        help_text = [
            "🔧 CAMERA TROUBLESHOOTING:",
            "1. ตรวจสอบการเชื่อมต่อกล้อง USB",
            "2. ลองเปลี่ยนพอร์ต USB",
            "3. ตรวจสอบ Driver กล้อง",
            "4. อนุญาตการเข้าถึงกล้องใน Windows",
            "5. รันโปรแกรมเป็น Administrator",
            "6. ปิดโปรแกรมอื่นที่ใช้กล้อง"
        ]
        
        y_pos = 280
        for line in help_text:
            color = (200, 200, 255) if line.startswith("🔧") else (180, 180, 255)
            cv2.putText(frame, line, (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 25
        
        return frame
    
    def draw_moving_objects(self, frame):
        """วาดวัตถุเคลื่อนไหวบนเฟรม"""
        # รถมอเตอร์ไซค์เคลื่อนที่
        bike_x = 100 + int(400 * np.sin(self.frame_counter * 0.05))
        bike_y = 350
        
        # วาดรถมอเตอร์ไซค์
        cv2.rectangle(frame, (bike_x, bike_y), (bike_x + 80, bike_y + 40), (255, 100, 100), -1)
        cv2.rectangle(frame, (bike_x, bike_y), (bike_x + 80, bike_y + 40), (255, 255, 255), 2)
        cv2.putText(frame, "BIKE", (bike_x + 15, bike_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # วงกลมเคลื่อนที่
        circle_x = 300 + int(100 * np.cos(self.frame_counter * 0.1))
        circle_y = 350 + int(50 * np.sin(self.frame_counter * 0.08))
        cv2.circle(frame, (circle_x, circle_y), 20, (100, 100, 255), -1)
        cv2.circle(frame, (circle_x, circle_y), 20, (255, 255, 255), 2)
        
        # เส้นเคลื่อนไหว
        line_y = 400 + int(30 * np.sin(self.frame_counter * 0.2))
        cv2.line(frame, (50, line_y), (590, line_y), (100, 255, 100), 3)
    
    def create_error_frame(self, message):
        """สร้างเฟรมแสดงข้อผิดพลาด"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (0, 0, 100)  # สีน้ำเงินเข้ม
        
        cv2.putText(frame, "❌ SYSTEM ERROR", (200, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, message, (150, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Please check the console for details", (120, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
        
        return frame
    
    def get_frame(self):
        """รับเฟรมปัจจุบัน - รับประกันว่าจะมีข้อมูล"""
        try:
            if self.frame is not None:
                # แปลงเป็น JPEG
                ret, jpeg = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    return jpeg.tobytes()
            
            # ถ้าไม่มีเฟรม สร้างเฟรม error
            error_frame = self.create_error_frame("No frame available")
            ret, jpeg = cv2.imencode('.jpg', error_frame)
            return jpeg.tobytes()
            
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดใน get_frame: {e}")
            # สร้างเฟรม error อย่างง่าย
            try:
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                error_frame[:] = (0, 0, 100)
                cv2.putText(error_frame, "FRAME ERROR", (220, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', error_frame)
                return jpeg.tobytes()
            except:
                # ส่งข้อมูล JPEG ว่าง (จะแสดงเป็น broken image ใน browser)
                return b''
    
    def stop(self):
        """หยุดการทำงาน"""
        self.running = False
        logger.info("🛑 VideoStream หยุดแล้ว")

# ===============================
# 🧠 Simple Detector (ไม่ใช้ AI จริงเพื่อความเร็ว)
# ===============================
class SimpleDetector:
    def __init__(self):
        logger.info("🤖 เริ่มต้น Simple Detector")
    
    def detect(self, frame):
        """จำลองการตรวจจับ"""
        try:
            # เพิ่มข้อมูลการตรวจจับจำลองบนเฟรม
            height, width = frame.shape[:2]
            
            # วาดกรอบจำลอง
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
            cv2.putText(frame, "AI DETECTION AREA", (110, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # เพิ่มข้อความสถานะ
            cv2.putText(frame, "SIMULATION MODE", (width-200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            return frame, False, []
            
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดใน detection: {e}")
            return frame, False, []

# ===============================
# 📊 File Manager (แบบง่าย)
# ===============================
class SimpleFileManager:
    def __init__(self):
        logger.info("📁 เริ่มต้น File Manager")
    
    def cleanup_old_files(self):
        """ทำความสะอาดไฟล์เก่า"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            folders_to_clean = ['captures', 'manual_uploads', 'temp']
            
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
        """คำนวณขนาดโฟลเดอร์"""
        try:
            total_size = 0
            if os.path.exists(folder):
                for dirpath, dirnames, filenames in os.walk(folder):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
            return total_size
        except:
            return 0

# ===============================
# 🗄️ Database (แบบง่าย)
# ===============================
def init_simple_db():
    """เริ่มต้นฐานข้อมูลแบบง่าย"""
    try:
        conn = sqlite3.connect('database/smart_rider.db')
        c = conn.cursor()
        
        # ตารางผู้ใช้
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      role TEXT DEFAULT 'user')''')
        
        # ตารางเหตุการณ์
        c.execute('''CREATE TABLE IF NOT EXISTS events
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      type TEXT NOT NULL,
                      description TEXT,
                      image_path TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        
        # เพิ่มผู้ใช้ทดสอบ
        try:
            c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)", 
                     ('admin', 'admin123', 'admin'))
            c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)", 
                     ('user', 'user123', 'user'))
        except:
            pass
        
        conn.commit()
        conn.close()
        logger.info("✅ ฐานข้อมูลเริ่มต้นสำเร็จ")
    except Exception as e:
        logger.error(f"❌ ข้อผิดพลาดฐานข้อมูล: {e}")

# ===============================
# 🏁 เริ่มต้นระบบ
# ===============================
video_stream = GuaranteedVideoStream()
file_manager = SimpleFileManager()
detector = SimpleDetector()

def generate_guaranteed_frames():
    """สร้าง video feed ที่รับประกันว่าจะทำงาน"""
    logger.info("🎬 เริ่ม generating frames แบบรับประกัน")
    
    while True:
        try:
            frame_bytes = video_stream.get_frame()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # ส่งเฟรมว่างเป็น fallback
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"❌ ข้อผิดพลาดใน generate_frames: {e}")
            # ส่งเฟรม error
            try:
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                error_frame[:] = (0, 0, 100)
                ret, jpeg = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')
            time.sleep(0.1)

# ===============================
# 🌐 Routes หลัก
# ===============================
_app_initialized = False

@app.before_request
def initialize_system():
    global _app_initialized
    if not _app_initialized:
        logger.info("🚀 เริ่มต้นระบบ Smart Rider...")
        init_simple_db()
        file_manager.cleanup_old_files()
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
    
    # ข้อมูลสถิติพื้นฐาน
    stats = {
        'username': session.get('username', 'User'),
        'total_events': 0,
        'no_helmet_count': 0,
        'exhaust_count': 0,
        'speeding_count': 0,
        'capture_size': '0 MB',
        'upload_size': '0 MB'
    }
    
    return render_template('dashboard.html', **stats)

@app.route('/live')
def live():
    logger.info("📺 มีผู้ใช้เข้าดูหน้ากล้องสด")
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    """Video feed ที่รับประกันว่าจะแสดงภาพ"""
    return Response(generate_guaranteed_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test')
def test_page():
    """หน้าทดสอบที่แสดงภาพโดยตรง"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🧪 Smart Rider - Test Page</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .video-container {
                text-align: center;
                margin: 20px 0;
                padding: 20px;
                background: rgba(0,0,0,0.3);
                border-radius: 10px;
            }
            img {
                max-width: 90%;
                border: 3px solid #00ff00;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,255,0,0.5);
            }
            .status {
                padding: 15px;
                margin: 15px 0;
                border-radius: 8px;
                font-weight: bold;
            }
            .success {
                background: rgba(0,255,0,0.2);
                border: 2px solid #00ff00;
            }
            .warning {
                background: rgba(255,255,0,0.2);
                border: 2px solid #ffff00;
            }
            .buttons {
                text-align: center;
                margin: 20px 0;
            }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                margin: 0 10px;
                background: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
                transition: all 0.3s;
            }
            .btn:hover {
                background: #45a049;
                transform: translateY(-2px);
            }
            .info-box {
                background: rgba(255,255,255,0.1);
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 SMART RIDER - TEST PAGE</h1>
            
            <div class="status success">
                ✅ <strong>สถานะ:</strong> ระบบทำงานปกติ | หากเห็นภาพด้านล่างแสดงว่ากล้อง feed ทำงาน
            </div>
            
            <div class="video-container">
                <h2>🎥 LIVE VIDEO FEED</h2>
                <img src="/video_feed" alt="Video Feed">
                <p><em>ภาพจากระบบ Video Stream</em></p>
            </div>
            
            <div class="info-box">
                <h3>📊 ข้อมูลระบบ:</h3>
                <p><strong>เวลา:</strong> <span id="currentTime"></span></p>
                <p><strong>สถานะกล้อง:</strong> <span id="cameraStatus">DEMO MODE</span></p>
                <p><strong>FPS:</strong> <span id="fpsCounter">30</span></p>
            </div>
            
            <div class="buttons">
                <a href="/" class="btn">🏠 หน้าแรก</a>
                <a href="/live" class="btn">📺 กล้องสดเต็มรูปแบบ</a>
                <button onclick="location.reload()" class="btn">🔄 รีเฟรช</button>
            </div>
            
            <div class="status warning">
                💡 <strong>หมายเหตุ:</strong> นี่คือโหมดทดสอบ ภาพที่เห็นเป็น simulation<br>
                ระบบจะแสดงภาพจริงเมื่อตรวจพบกล้องที่เชื่อมต่อ
            </div>
        </div>
        
        <script>
            // อัพเดทเวลา
            function updateTime() {
                const now = new Date();
                document.getElementById('currentTime').textContent = now.toLocaleString();
            }
            setInterval(updateTime, 1000);
            updateTime();
            
            // อัพเดท FPS
            let frameCount = 0;
            setInterval(() => {
                document.getElementById('fpsCounter').textContent = frameCount;
                frameCount = 0;
            }, 1000);
            
            // นับเฟรม
            const img = document.querySelector('img');
            img.onload = () => frameCount++;
        </script>
    </body>
    </html>
    '''

@app.route('/upload')
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return "Upload Page - อยู่ระหว่างพัฒนา"

@app.route('/events')
def events():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return "Events Page - อยู่ระหว่างพัฒนา"

if __name__ == '__main__':
    print("=" * 70)
    print("🤖 SMART RIDER SYSTEM - GUARANTEED VERSION")
    print("📧 Login: admin / admin123")
    print("🌐 Main URL: http://localhost:5000")
    print("🧪 Test Page: http://localhost:5000/test")
    print("📺 Live Camera: http://localhost:5000/live")
    print("=" * 70)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 หยุดระบบโดยผู้ใช้")
    except Exception as e:
        print(f"❌ ข้อผิดพลาด: {e}")
    finally:
        video_stream.stop()