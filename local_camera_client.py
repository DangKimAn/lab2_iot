import cv2
import time
import requests
import json
import os
import threading
from dotenv import load_dotenv

load_dotenv()

# Thay đổi URL này thành IP của EC2 khi đưa lên mây
SERVER_IP = os.environ.get('SERVER_IP', '13.213.28.1')
SERVER_URL = f'http://{SERVER_IP}:5000/api/v2/detect'

# Biến dùng chung giữa 2 luồng
latest_frame_to_send = None
latest_data = {"boxes": [], "latency_text": "Dang cho du lieu tu Cloud..."}
lock = threading.Lock()

def network_worker():
    global latest_frame_to_send, latest_data
    while True:
        frame_copy = None
        with lock:
            if latest_frame_to_send is not None:
                frame_copy = latest_frame_to_send.copy()
                latest_frame_to_send = None # Xóa sau khi lấy để tránh gửi trùng
        
        if frame_copy is None:
            time.sleep(0.01)
            continue
            
        t_start_e2e = time.time()
        # Nén ảnh thành JPEG để giảm băng thông
        ret_enc, buffer = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret_enc:
            continue
            
        image_bytes = buffer.tobytes()
        
        try:
            files = {'image': ('frame.jpg', image_bytes, 'image/jpeg')}
            response = requests.post(SERVER_URL, files=files, timeout=2.0)
            t_end_e2e = time.time()
            
            if response.status_code == 200:
                data = response.json()
                inference_ms = data.get("inference_time_ms", 0.0)
                detections = data.get("detections", 0)
                log_id = data.get("log_id")
                boxes = data.get("boxes", [])
                
                total_latency_ms = (t_end_e2e - t_start_e2e) * 1000
                network_latency_ms = total_latency_ms - inference_ms
                
                latency_text = f"E2E: {total_latency_ms:.1f}ms | AI: {inference_ms:.1f}ms | Net: {network_latency_ms:.1f}ms"
                
                # Cập nhật kết quả mới nhất cho luồng Camera vẽ lên màn hình
                with lock:
                    latest_data["boxes"] = boxes
                    latest_data["latency_text"] = latency_text
                
                print(f"[Latency] Toan trinh: {total_latency_ms:.1f}ms | Mang Local<->Cloud: {network_latency_ms:.1f}ms | Mat: {detections}")
                
                if log_id is not None:
                    update_url = SERVER_URL.replace('/detect', '/update_latency')
                    payload = {
                        "log_id": log_id,
                        "network_latency_ms": network_latency_ms,
                        "e2e_latency_ms": total_latency_ms
                    }
                    try:
                        requests.post(update_url, json=payload, timeout=2.0)
                    except:
                        pass
        except Exception as e:
            pass
            
        # Giới hạn tốc độ gửi lên server (tránh spam gây sập EC2)
        time.sleep(0.05)

def main():
    global latest_frame_to_send
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera.")
        return

    print(f"Đang gửi dữ liệu tới Server: {SERVER_URL}")
    print("Nhấn 'q' để thoát.")

    # Khởi động luồng mạng chạy ngầm (Asynchronous)
    net_thread = threading.Thread(target=network_worker, daemon=True)
    net_thread.start()

    colors_map = {
        "wrong mask": (0, 255, 255),
        "mask": (0, 255, 0),
        "no mask": (0, 0, 255)
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc từ camera.")
            break

        # Đẩy ảnh mới nhất cho luồng mạng gửi đi (chỉ đẩy nếu luồng mạng đang rảnh rỗi)
        with lock:
            if latest_frame_to_send is None:
                latest_frame_to_send = frame.copy()
            
            # Copy data mới nhất từ server để vẽ
            boxes_to_draw = latest_data["boxes"]
            latency_text = latest_data["latency_text"]

        # Vẽ box nhận được từ server lên frame
        for b in boxes_to_draw:
            x1, y1, x2, y2 = b.get("box", [0,0,0,0])
            name = b.get("name", "Unknown")
            conf = b.get("conf", 0.0)
            track_id = b.get("track_id", "None")
            
            color = colors_map.get(name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {name} {int(conf * 100)}%"
            cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Hiển thị độ trễ lên góc màn hình
        cv2.rectangle(frame, (10, 10), (10 + 600, 10 + 30), (0, 0, 0), -1)
        cv2.putText(frame, latency_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Hiển thị ảnh tức thời, không bị block bởi requests.post
        try:
            cv2.imshow("Local Camera Client", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            pass

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

if __name__ == '__main__':
    main()
