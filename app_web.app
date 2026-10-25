'''
import os
import cv2
import math
import time
import uuid       
import pytz      
import threading
import boto3
from datetime import datetime 
from dotenv import load_dotenv
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, request, make_response

load_dotenv()
HTML_TEMPLATE = os.getenv("HTML_TEMPLATE", os.getenv("TEMPLATE_FILE", "index.html"))

VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

app = Flask(__name__)

model = YOLO("mask_model.pt")
classNames = ["wrong mask", "mask", "no mask"] 

# Màu sắc cho khung (chuẩn BGR của OpenCV)
colors = [
    (0, 255, 255),   
    (0, 255, 0),    
    (0, 0, 255)    
]

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)
BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')


def upload_to_s3_async(frame_bytes, object_name):
    def upload_task():
        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME, 
                Key=object_name, 
                Body=frame_bytes, 
                ContentType='image/jpeg',
                # ACL='public-read' 
            )
            print(f"--> [S3] Upload thành công: {object_name}")
        except Exception as e:
            print(f"--> [S3] Lỗi upload: {e}")
            
    # Chạy hàm upload trong một Thread độc lập
    thread = threading.Thread(target=upload_task)
    thread.start()


detection_logs = []

def generate_frames(camera_url):
    src = int(camera_url) if str(camera_url).isdigit() else camera_url
    cap = cv2.VideoCapture(src)
    
    if not cap.isOpened():
        print(f"Không thể kết nối tới camera: {camera_url}")
        return

    CONF_THRESHOLD = 0.80      
    STABLE_DURATION = 3.0    
    LOST_TOLERANCE = 1.0     

    last_notified_status = detection_logs[0]["status"] if detection_logs else None
    candidate_status = None
    candidate_start_time = 0.0
    candidate_last_seen = 0.0

    try:
        while True:
            success, img = cap.read()
            if not success:
                time.sleep(0.1)
                continue

            current_time = time.time()
            results = model(img, stream=True, verbose=False)

            best_status = None
            best_conf = 0.0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    name = classNames[cls] if cls < len(classNames) else "Unknown"
                    color = colors[cls] if cls < len(colors) else (255, 255, 255)

                    if conf > 0.5:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, f'{name} {int(conf * 100)}%', (x1, max(30, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if conf >= CONF_THRESHOLD and conf > best_conf:
                        best_conf = conf
                        best_status = name

            if not detection_logs:
                last_notified_status = None

            if best_status is not None:
                if best_status == candidate_status:
                    candidate_last_seen = current_time

                    if (current_time - candidate_start_time) >= STABLE_DURATION:
                        if candidate_status != last_notified_status:
                            last_notified_status = candidate_status

                            ret_img, buffer = cv2.imencode('.jpg', img)
                            s3_url = None
                            
                            if ret_img:
                                # <-- Sửa lỗi VN_TZ và uuid ở đây
                                date_str = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M%S")
                                safe_status = candidate_status.replace(" ", "_")
                                file_name = f"mask_logs_v2/{date_str}_{safe_status}_{uuid.uuid4().hex[:6]}.jpg"
                                
                                frame_bytes = buffer.tobytes()
                                upload_to_s3_async(frame_bytes, file_name)
                                
                                # s3_url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{file_name}"

                                s3_url = f"s3://{BUCKET_NAME}/mask_logs_v2/{file_name}"
                            
                            timestamp = time.strftime("%H:%M:%S")
                            log_entry = {
                                "time": timestamp,
                                "status": candidate_status,
                                "confidence": f"{best_conf * 100:.0f}%",
                                "raw_time": current_time,
                                "image_url": s3_url # <-- Cập nhật url ảnh vào log
                            }
                            
                            detection_logs.insert(0, log_entry)
                            if len(detection_logs) > 50:
                                detection_logs.pop()
                            print(f"[{timestamp}] [THÔNG BÁO] Thay đổi trạng thái -> {candidate_status} ({best_conf * 100:.0f}%) duy trì >= 3s")
                else:
                    candidate_status = best_status
                    candidate_start_time = current_time
                    candidate_last_seen = current_time
            else:
                if candidate_status is not None and (current_time - candidate_last_seen > LOST_TOLERANCE):
                    candidate_status = None
                    candidate_start_time = 0.0

            overlay_text = f"Status: {last_notified_status if last_notified_status else 'None'}"
            if candidate_status and candidate_status != last_notified_status:
                elapsed = min(STABLE_DURATION, current_time - candidate_start_time)
                overlay_text += f" | Pending: {candidate_status} ({elapsed:.1f}s/{STABLE_DURATION:.0f}s)"

            (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (10, 10), (10 + tw + 10, 10 + th + 14), (0, 0, 0), -1)
            text_color = (0, 255, 255) if "Pending" in overlay_text else (0, 255, 0)
            cv2.putText(img, overlay_text, (15, 10 + th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

@app.route('/')
def index():
    global detection_logs
    detection_logs.clear()
    
    response = make_response(render_template(HTML_TEMPLATE))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/video_feed')
def video_feed():
    ip_address = request.args.get('ip', '')
    
    if not ip_address:
        return "Chưa cung cấp IP", 400
        
    if ip_address == "0" or ip_address.lower() == "webcam":
        camera_url = 0
    elif not ip_address.startswith("http"):
        camera_url = f"http://{ip_address}/stream"
    else:
        camera_url = ip_address
        
    print(f"Đang thử kết nối tới: {camera_url}")
    return Response(generate_frames(camera_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    return jsonify(detection_logs)

@app.route('/api/logs/clear', methods=['GET', 'POST'])
def clear_logs():
    global detection_logs
    detection_logs.clear()
    return jsonify({"status": "success", "message": "Đã xóa toàn bộ log tạm thời"})

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port, debug=True)
'''

'''

import os
import cv2
import math
import time
import uuid
import pytz
import threading
import boto3
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, request, make_response
from flask_sqlalchemy import SQLAlchemy  # <-- Import SQLAlchemy

# Tải cấu hình từ file .env
load_dotenv()
HTML_TEMPLATE = os.getenv("HTML_TEMPLATE", os.getenv("TEMPLATE_FILE", "index.html"))

# Cấu hình Múi giờ Việt Nam
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

app = Flask(__name__)

# --- CẤU HÌNH SUPABASE (POSTGRESQL) ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- KHAI BÁO BẢNG DATABASE ---
class DetectionLog(db.Model):
    __tablename__ = 'detection_logs_v2'
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.String(10), nullable=False)
    image_url = db.Column(db.String(255), nullable=True)  # Cột lưu link S3
    inference_time_ms = db.Column(db.Float, nullable=True) # Thời gian suy luận AI
    network_latency_ms = db.Column(db.Float, nullable=True)
    e2e_latency_ms = db.Column(db.Float, nullable=True)
    supabase_latency_ms = db.Column(db.Float, nullable=True)
    telegram_latency_ms = db.Column(db.Float, nullable=True)

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(VN_TZ).replace(tzinfo=None))

    def to_dict(self):
        return {
            "id": self.id,
            "time": self.created_at.strftime("%H:%M:%S"),
            "status": self.status,
            "confidence": self.confidence,
            "image_url": self.image_url,
            "date": self.created_at.strftime("%d/%m/%Y")
        }

model = YOLO("mask_model.pt")
classNames = ["wrong mask", "mask", "no mask"] 

# Màu sắc cho khung
colors = [
    (0, 255, 255),   # Vàng cho "Đeo sai"
    (0, 255, 0),     # Xanh lá cho "Đeo đúng"
    (0, 0, 255)      # Đỏ cho "Không đeo"
]

# --- CẤU HÌNH S3 CLIENT ---
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)
BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')

def upload_to_s3_async(frame_bytes, object_name):
    def upload_task():
        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME, 
                Key=object_name, 
                Body=frame_bytes, 
                ContentType='image/jpeg',
                # ACL='public-read' 
            )
            print(f"--> [S3] Upload thành công: {object_name}")
        except Exception as e:
            print(f"--> [S3] Lỗi upload: {e}")
            
    thread = threading.Thread(target=upload_task)
    thread.start()

# --- BIẾN TOÀN CỤC LƯU TRẠNG THÁI HIỆN TẠI ---
last_notified_status = None

# Khởi tạo bảng và lấy trạng thái gần nhất khi app chạy
with app.app_context():
    db.create_all()
    latest_log = DetectionLog.query.order_by(DetectionLog.created_at.desc()).first()
    if latest_log:
        last_notified_status = latest_log.status

def generate_frames(camera_url):
    global last_notified_status
    src = int(camera_url) if str(camera_url).isdigit() else camera_url
    cap = cv2.VideoCapture(src)
    
    if not cap.isOpened():
        print(f"Không thể kết nối tới camera: {camera_url}")
        return

    CONF_THRESHOLD = 0.80      
    STABLE_DURATION = 3.0    
    LOST_TOLERANCE = 1.0     

    candidate_status = None
    candidate_start_time = 0.0
    candidate_last_seen = 0.0

    try:
        while True:
            success, img = cap.read()
            if not success:
                time.sleep(0.1)
                continue

            current_time = time.time()
            results = model(img, stream=True, verbose=False)

            best_status = None
            best_conf = 0.0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    name = classNames[cls] if cls < len(classNames) else "Unknown"
                    color = colors[cls] if cls < len(colors) else (255, 255, 255)

                    if conf > 0.5:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, f'{name} {int(conf * 100)}%', (x1, max(30, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if conf >= CONF_THRESHOLD and conf > best_conf:
                        best_conf = conf
                        best_status = name

            if best_status is not None:
                if best_status == candidate_status:
                    candidate_last_seen = current_time

                    if (current_time - candidate_start_time) >= STABLE_DURATION:
                        if candidate_status != last_notified_status:
                            last_notified_status = candidate_status

                            ret_img, buffer = cv2.imencode('.jpg', img)
                            s3_url = None
                            
                            if ret_img:
                                date_str = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M%S")
                                safe_status = candidate_status.replace(" ", "_")
                                file_name = f"mask_logs_v2/{date_str}_{safe_status}_{uuid.uuid4().hex[:6]}.jpg"
                                
                                frame_bytes = buffer.tobytes()
                                upload_to_s3_async(frame_bytes, file_name)
                            
                                s3_url = f"s3://{BUCKET_NAME}/mask_logs_v2/{file_name}"
                            # --- GHI VÀO SUPABASE ---
                            with app.app_context():
                                new_log = DetectionLog(
                                    status=candidate_status,
                                    confidence=f"{best_conf * 100:.0f}%",
                                    image_url=s3_url
                                )
                                db.session.add(new_log)
                                db.session.commit()

                            timestamp = time.strftime("%H:%M:%S")
                            print(f"[{timestamp}] [THÔNG BÁO] Thay đổi trạng thái -> {candidate_status} ({best_conf * 100:.0f}%) lưu DB thành công!")
                else:
                    candidate_status = best_status
                    candidate_start_time = current_time
                    candidate_last_seen = current_time
            else:
                if candidate_status is not None and (current_time - candidate_last_seen > LOST_TOLERANCE):
                    candidate_status = None
                    candidate_start_time = 0.0

            overlay_text = f"Status: {last_notified_status if last_notified_status else 'None'}"
            if candidate_status and candidate_status != last_notified_status:
                elapsed = min(STABLE_DURATION, current_time - candidate_start_time)
                overlay_text += f" | Pending: {candidate_status} ({elapsed:.1f}s/{STABLE_DURATION:.0f}s)"

            (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (10, 10), (10 + tw + 10, 10 + th + 14), (0, 0, 0), -1)
            text_color = (0, 255, 255) if "Pending" in overlay_text else (0, 255, 0)
            cv2.putText(img, overlay_text, (15, 10 + th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

@app.route('/')
def index():
    # Không xóa log khi F5 nữa vì log giờ nằm an toàn trên Supabase
    response = make_response(render_template(HTML_TEMPLATE))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

def generate_esp32_frames():
    global latest_esp32_frame
    while True:
        if latest_esp32_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_esp32_frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    ip_address = request.args.get('ip', '')
    if not ip_address:
        return "Chưa cung cấp IP", 400
        
    if ip_address.lower() == "esp32":
        return Response(generate_esp32_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
    if ip_address == "0" or ip_address.lower() == "webcam":
        camera_url = 0
    elif not ip_address.startswith("http"):
        camera_url = f"http://{ip_address}/stream"
    else:
        camera_url = ip_address
        
    print(f"Đang thử kết nối tới: {camera_url}")
    return Response(generate_frames(camera_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    # Kéo 50 log mới nhất từ Supabase trả về cho Web
    logs = DetectionLog.query.order_by(DetectionLog.created_at.desc()).limit(50).all()
    return jsonify([log.to_dict() for log in logs])

@app.route('/api/logs/clear', methods=['GET', 'POST'])
def clear_logs():
    # Cẩn thận: API này sẽ xóa SẠCH bảng trên Supabase
    global last_notified_status
    last_notified_status = None
    try:
        num_rows_deleted = db.session.query(DetectionLog).delete()
        db.session.commit()
        return jsonify({"status": "success", "message": f"Đã xóa {num_rows_deleted} record trong DB"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port, debug=True)


'''


import os
import cv2
import math
import time
import uuid
import pytz
import threading
import boto3
import requests
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, request, make_response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.pool import NullPool

# Tải cấu hình từ file .env
load_dotenv()
HTML_TEMPLATE = os.getenv("HTML_TEMPLATE", os.getenv("TEMPLATE_FILE", "index.html"))

# Cấu hình Múi giờ Việt Nam
VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

app = Flask(__name__)

# --- CẤU HÌNH SUPABASE (POSTGRESQL) ---
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Tắt chế độ Pool của SQLAlchemy, nhường quyền quản lý connection cho Supabase PgBouncer
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {"poolclass": NullPool} 

db = SQLAlchemy(app)

# --- KHAI BÁO BẢNG DATABASE ---
class DetectionLog(db.Model):
    __tablename__ = 'detection_logs_v2'
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.String(10), nullable=False)
    image_url = db.Column(db.String(255), nullable=True)  # Cột lưu link S3
    inference_time_ms = db.Column(db.Float, nullable=True) # Thời gian suy luận AI
    network_latency_ms = db.Column(db.Float, nullable=True)
    e2e_latency_ms = db.Column(db.Float, nullable=True)
    supabase_latency_ms = db.Column(db.Float, nullable=True)
    telegram_latency_ms = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(VN_TZ).replace(tzinfo=None))

    def to_dict(self):
        return {
            "id": self.id,
            "time": self.created_at.strftime("%H:%M:%S"),
            "status": self.status,
            "confidence": self.confidence,
            "image_url": self.image_url,
            "date": self.created_at.strftime("%d/%m/%Y")
        }

model = YOLO("mask_model.pt")
classNames = ["wrong mask", "mask", "no mask"] 

# Màu sắc cho khung
colors = [
    (0, 255, 255),   # Vàng cho "Đeo sai" (wrong mask)
    (0, 255, 0),     # Xanh lá cho "Đeo đúng" (mask)
    (0, 0, 255)      # Đỏ cho "Không đeo" (no mask)
]

# --- CẤU HÌNH AWS S3 ---
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)
BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')

def upload_to_s3_async(frame_bytes, object_name):
    def upload_task():
        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME, 
                Key=object_name, 
                Body=frame_bytes, 
                ContentType='image/jpeg',
                # ACL='public-read' 
            )
            print(f"--> [S3] Upload thành công: {object_name}")
        except Exception as e:
            print(f"--> [S3] Lỗi upload: {e}")
            
    thread = threading.Thread(target=upload_task)
    thread.start()

# --- CẤU HÌNH TELEGRAM ---
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def send_telegram_alert_async(status, confidence, image_bytes, log_id=None, db_latency_ms=None):
    def telegram_task():
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            print("--> [Telegram] Bỏ qua: Chưa cấu hình Token hoặc Chat ID trong file .env")
            return
            
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        
        current_time_str = datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M:%S")
        caption = (
            f"🚨 <b>CẢNH BÁO NHẬN DIỆN</b> 🚨\n"
            f"⏱ <b>Thời gian:</b> {current_time_str}\n"
            f"⚠️ <b>Trạng thái:</b> {status}\n"
            f"📊 <b>Độ tin cậy:</b> {confidence}"
        )
        
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML"
        }
        
        files = {
            "photo": ("alert.jpg", image_bytes, "image/jpeg")
        }
        
        try:
            t_start = time.time()
            response = requests.post(url, data=payload, files=files)
            t_end = time.time()
            telegram_latency_ms = (t_end - t_start) * 1000
            
            if response.status_code == 200:
                print(f"--> [Telegram] Đã gửi thông báo cảnh báo thành công! (Thời gian trễ: {telegram_latency_ms:.1f}ms)")
                
                # Cập nhật Supabase & Telegram latency vào DB nếu có log_id
                if log_id is not None:
                    with app.app_context():
                        log = DetectionLog.query.get(log_id)
                        if log:
                            log.supabase_latency_ms = db_latency_ms
                            log.telegram_latency_ms = round(telegram_latency_ms, 2)
                            db.session.commit()
            else:
                print(f"--> [Telegram] Lỗi gửi tin nhắn: {response.text}")
        except Exception as e:
            print(f"--> [Telegram] Lỗi kết nối API: {e}")

    thread = threading.Thread(target=telegram_task)
    thread.start()


# --- BIẾN TOÀN CỤC LƯU TRẠNG THÁI HIỆN TẠI ---
last_notified_status = None

# Khởi tạo bảng và lấy trạng thái gần nhất khi app chạy
with app.app_context():
    db.create_all()
    latest_log = DetectionLog.query.order_by(DetectionLog.created_at.desc()).first()
    if latest_log:
        last_notified_status = latest_log.status

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

class VideoCaptureThread:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        
        if self.ret:
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
            
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.01)
                
    def read(self):
        return self.ret, self.frame.copy() if self.frame is not None else None
        
    def release(self):
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()
        
    def isOpened(self):
        return self.cap.isOpened()

def generate_frames(camera_url):
    global last_notified_status
    src = int(camera_url) if str(camera_url).isdigit() else camera_url
    cap = VideoCaptureThread(src)
    
    if not cap.isOpened():
        print(f"Không thể kết nối tới camera: {camera_url}")
        return

    CONF_THRESHOLD = 0.80      
    WINDOW_SIZE = 10
    MIN_VOTES = 5
    VOTE_THRESHOLD = 0.7
    LOST_TOLERANCE = 1.0     
    # Dictionary theo dõi trạng thái từng khuôn mặt:
    # face_tracks[track_id] = {"status", "confidence", "box", "history", "last_seen", "last_notified_status", "notified"}
    face_tracks = {}
    next_custom_id = 1000
    latest_detected_faces = []
    
    def ai_worker(frame_to_track, current_time):
        nonlocal latest_detected_faces, next_custom_id, face_tracks
        global last_notified_status

        t_start_inf = time.time()
        results = model.track(frame_to_track, persist=True, verbose=False)
        t_end_inf = time.time()
        inference_time_ms = (t_end_inf - t_start_inf) * 1000
        detected = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = classNames[cls] if cls < len(classNames) else "Unknown"
                color = colors[cls] if cls < len(colors) else (255, 255, 255)
                track_id = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else None

                detected.append({
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "cls": cls,
                    "name": name,
                    "color": color,
                    "track_id": track_id
                })

        for det in detected:
            x1, y1, x2, y2 = det["box"]
            track_id = det["track_id"]
            if track_id is None:
                best_iou = 0.0
                matched_id = None
                for fid, f_data in face_tracks.items():
                    iou = calculate_iou((x1, y1, x2, y2), f_data["box"])
                    if iou > 0.3 and iou > best_iou:
                        best_iou = iou
                        matched_id = fid
                if matched_id is not None:
                    track_id = matched_id
                else:
                    next_custom_id += 1
                    track_id = next_custom_id
                det["track_id"] = track_id

            if det["conf"] >= CONF_THRESHOLD:
                name = det["name"]
                conf = det["conf"]
                if track_id not in face_tracks:
                    face_tracks[track_id] = {
                        "status": name,
                        "confidence": conf,
                        "box": (x1, y1, x2, y2),
                        "history": [name],
                        "last_seen": current_time,
                        "last_notified_status": None,
                        "notified": False
                    }
                else:
                    track_info = face_tracks[track_id]
                    track_info["box"] = (x1, y1, x2, y2)
                    track_info["last_seen"] = current_time
                    track_info["confidence"] = conf
                    
                    track_info["history"].append(name)
                    if len(track_info["history"]) > WINDOW_SIZE:
                        track_info["history"].pop(0)

                    if len(track_info["history"]) >= MIN_VOTES:
                        from collections import Counter
                        vote_counts = Counter(track_info["history"])
                        most_common_status, most_common_count = vote_counts.most_common(1)[0]
                        
                        if most_common_count / len(track_info["history"]) >= VOTE_THRESHOLD:
                            track_info["status"] = most_common_status
                            should_notify = (not track_info["notified"]) or (most_common_status != track_info["last_notified_status"])
                            if should_notify:
                                notify_reason = "Đổi khuôn mặt" if not track_info["notified"] else "Đổi trạng thái"
                                track_info["notified"] = True
                                track_info["last_notified_status"] = most_common_status
                                last_notified_status = most_common_status

                                ret_img, buffer = cv2.imencode('.jpg', frame_to_track)
                                s3_url = None
                                frame_bytes = None
                                
                                if ret_img:
                                    date_str = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M%S")
                                    safe_status = most_common_status.replace(" ", "_")
                                    file_name = f"mask_logs_v2/{date_str}_{safe_status}_{uuid.uuid4().hex[:6]}.jpg"
                                    frame_bytes = buffer.tobytes()
                                    upload_to_s3_async(frame_bytes, file_name)
                                    s3_url = f"s3://{BUCKET_NAME}/mask_logs_v2/{file_name}"
                                
                                with app.app_context():
                                    new_log = DetectionLog(
                                        status=most_common_status,
                                        confidence=f"{conf * 100:.0f}%",
                                        image_url=s3_url,
                                        inference_time_ms=round(inference_time_ms, 2)
                                    )
                                    t_start_db = time.time()
                                    db.session.add(new_log)
                                    db.session.commit()
                                    t_end_db = time.time()

                                timestamp = time.strftime("%H:%M:%S")
                                db_latency_ms = (t_end_db - t_start_db) * 1000
                                print(f"[{timestamp}] [THÔNG BÁO] [{notify_reason}] ID:{track_id} -> {most_common_status} ({conf * 100:.0f}%) đã lưu DB! (Trễ DB: {db_latency_ms:.1f}ms)")
                                
                                if frame_bytes:
                                    send_telegram_alert_async(
                                        status=f"{most_common_status} (ID:{track_id} - {notify_reason})", 
                                        confidence=f"{conf * 100:.0f}%", 
                                        image_bytes=frame_bytes,
                                        log_id=new_log.id,
                                        db_latency_ms=round(db_latency_ms, 2)
                                    )

        lost_ids = [fid for fid, f_data in face_tracks.items() if current_time - f_data["last_seen"] > LOST_TOLERANCE]
        for fid in lost_ids:
            del face_tracks[fid]
            
        latest_detected_faces = detected

    ai_thread = None

    try:
        while True:
            success, img = cap.read()
            if not success or img is None:
                time.sleep(0.01)
                continue

            current_time = time.time()

            if ai_thread is None or not ai_thread.is_alive():
                ai_thread = threading.Thread(target=ai_worker, args=(img.copy(), current_time))
                ai_thread.daemon = True
                ai_thread.start()

            # VẼ KHUNG CHO TẤT CẢ CÁC FRAME TỪ KẾT QUẢ GẦN NHẤT
            for det in latest_detected_faces:
                if det["conf"] > 0.5:
                    x1, y1, x2, y2 = det["box"]
                    cv2.rectangle(img, (x1, y1), (x2, y2), det["color"], 2)
                    label = f"ID:{det['track_id']} {det['name']} {int(det['conf'] * 100)}%"
                    cv2.putText(img, label, (x1, max(30, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, det["color"], 2)

            # Hiển thị thông tin trạng thái và đếm thời gian trên khung hình
            overlay_text = f"Status: {last_notified_status if last_notified_status else 'None'}"
            pending_items = []
            for fid, f_data in face_tracks.items():
                if (not f_data["notified"]) or (f_data.get("status") != f_data["last_notified_status"]):
                    history_len = len(f_data.get("history", []))
                    pending_items.append(f"ID:{fid} Voting... ({history_len}/{WINDOW_SIZE})")

            if pending_items:
                overlay_text += f" | Pending: {', '.join(pending_items)}"

            (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (10, 10), (10 + tw + 10, 10 + th + 14), (0, 0, 0), -1)
            text_color = (0, 255, 255) if pending_items else (0, 255, 0)
            cv2.putText(img, overlay_text, (15, 10 + th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

api_face_tracks = {}
api_next_custom_id = 1000
api_last_notified_status = None
latest_esp32_frame = None

@app.route('/api/v2/detect', methods=['POST'])
def api_v2_detect():
    import numpy as np
    from flask import request, jsonify
    
    global api_next_custom_id, api_last_notified_status, latest_esp32_frame
    
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
        
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Invalid image"}), 400
        
    current_time = time.time()
    
    # Bấm giờ Inference
    t_start = time.time()
    results = model.track(img, persist=True, verbose=False)
    t_end = time.time()
    inference_time_ms = (t_end - t_start) * 1000
    
    WINDOW_SIZE = 10
    MIN_VOTES = 5
    VOTE_THRESHOLD = 0.7
    CONF_THRESHOLD = 0.80
    LOST_TOLERANCE = 1.0

    detected = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = classNames[cls] if cls < len(classNames) else "Unknown"
            track_id = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else None
            detected.append({"box": (x1, y1, x2, y2), "conf": conf, "name": name, "track_id": track_id})

    created_log_id = None

    # Tracking & Logic
    for det in detected:
        x1, y1, x2, y2 = det["box"]
        track_id = det["track_id"]
        if track_id is None:
            best_iou = 0.0
            matched_id = None
            for fid, f_data in api_face_tracks.items():
                iou = calculate_iou((x1, y1, x2, y2), f_data["box"])
                if iou > 0.3 and iou > best_iou:
                    best_iou = iou
                    matched_id = fid
            if matched_id is not None:
                track_id = matched_id
            else:
                api_next_custom_id += 1
                track_id = api_next_custom_id
            det["track_id"] = track_id

        if det["conf"] >= CONF_THRESHOLD:
            name = det["name"]
            conf = det["conf"]
            
            # Vẽ box
            color = (255, 255, 255)
            if name == "wrong mask": color = (0, 255, 255)
            elif name == "mask": color = (0, 255, 0)
            elif name == "no mask": color = (0, 0, 255)
                
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {name} {int(conf * 100)}%"
            cv2.putText(img, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if track_id not in api_face_tracks:
                api_face_tracks[track_id] = {
                    "status": name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "history": [name],
                    "last_seen": current_time,
                    "last_notified_status": None,
                    "notified": False
                }
            else:
                track_info = api_face_tracks[track_id]
                track_info["box"] = (x1, y1, x2, y2)
                track_info["last_seen"] = current_time
                track_info["confidence"] = conf
                track_info["history"].append(name)
                
                if len(track_info["history"]) > WINDOW_SIZE:
                    track_info["history"].pop(0)

                if len(track_info["history"]) >= MIN_VOTES:
                    from collections import Counter
                    vote_counts = Counter(track_info["history"])
                    most_common_status, most_common_count = vote_counts.most_common(1)[0]
                    
                    if most_common_count / len(track_info["history"]) >= VOTE_THRESHOLD:
                        track_info["status"] = most_common_status
                        should_notify = (not track_info["notified"]) or (most_common_status != track_info["last_notified_status"])
                        if should_notify:
                            notify_reason = "Đổi khuôn mặt" if not track_info["notified"] else "Đổi trạng thái"
                            track_info["notified"] = True
                            track_info["last_notified_status"] = most_common_status
                            api_last_notified_status = most_common_status

                            ret_img, buffer = cv2.imencode('.jpg', img)
                            if ret_img:
                                date_str = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M%S")
                                safe_status = most_common_status.replace(" ", "_")
                                file_name = f"mask_logs_v2/{date_str}_{safe_status}_{uuid.uuid4().hex[:6]}.jpg"
                                frame_bytes = buffer.tobytes()
                                upload_to_s3_async(frame_bytes, file_name)
                                s3_url = f"s3://{BUCKET_NAME}/mask_logs_v2/{file_name}"
                            
                                with app.app_context():
                                    new_log = DetectionLog(
                                        status=most_common_status,
                                        confidence=f"{conf * 100:.0f}%",
                                        image_url=s3_url,
                                        inference_time_ms=round(inference_time_ms, 2)
                                    )
                                    t_start_db = time.time()
                                    db.session.add(new_log)
                                    db.session.commit()
                                    t_end_db = time.time()
                                    created_log_id = new_log.id
                                    
                                    db_latency_ms = (t_end_db - t_start_db) * 1000
                                    print(f"--> [Supabase] Đã lưu DB API thành công! (Thời gian trễ: {db_latency_ms:.1f}ms)")

                                send_telegram_alert_async(
                                    status=f"{most_common_status} (API - {notify_reason})", 
                                    confidence=f"{conf * 100:.0f}%", 
                                    image_bytes=frame_bytes,
                                    log_id=created_log_id,
                                    db_latency_ms=round(db_latency_ms, 2)
                                )

    lost_ids = [fid for fid, f_data in api_face_tracks.items() if current_time - f_data["last_seen"] > LOST_TOLERANCE]
    for fid in lost_ids:
        del api_face_tracks[fid]

    # Lưu frame mới nhất để phát lên web
    latest_esp32_frame = img.copy()

    return jsonify({
        "success": True,
        "inference_time_ms": round(inference_time_ms, 2),
        "detections": len(detected),
        "log_id": created_log_id,
        "boxes": detected
    })

@app.route('/api/v2/update_latency', methods=['POST'])
def api_v2_update_latency():
    from flask import request, jsonify
    data = request.get_json()
    if not data or 'log_id' not in data:
        return jsonify({"error": "Missing log_id"}), 400
        
    log_id = data['log_id']
    network_latency = data.get('network_latency_ms')
    e2e_latency = data.get('e2e_latency_ms')
    
    with app.app_context():
        log = DetectionLog.query.get(log_id)
        if log:
            if network_latency is not None:
                log.network_latency_ms = round(network_latency, 2)
            if e2e_latency is not None:
                log.e2e_latency_ms = round(e2e_latency, 2)
            db.session.commit()
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Log not found"}), 404

@app.route('/')
def index():
    response = make_response(render_template(HTML_TEMPLATE))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/video_feed')
def video_feed():
    ip_address = request.args.get('ip', '')
    if not ip_address:
        return "Chưa cung cấp IP", 400
        
    if ip_address == "0" or ip_address.lower() == "webcam":
        camera_url = 0
    elif not ip_address.startswith("http"):
        camera_url = f"http://{ip_address}/stream"
    else:
        camera_url = ip_address
        
    print(f"Đang thử kết nối tới: {camera_url}")
    return Response(generate_frames(camera_url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    logs = DetectionLog.query.order_by(DetectionLog.created_at.desc()).limit(50).all()
    return jsonify([log.to_dict() for log in logs])

@app.route('/api/logs/clear', methods=['GET', 'POST'])
def clear_logs():
    global last_notified_status
    last_notified_status = None
    try:
        num_rows_deleted = db.session.query(DetectionLog).delete()
        db.session.commit()
        return jsonify({"status": "success", "message": f"Đã xóa {num_rows_deleted} record trong DB"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port, debug=True)
