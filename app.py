import cv2
import math
import time
from ultralytics import YOLO

# 1. Tải mô hình YOLOv8 đã nhận diện khẩu trang
model = YOLO("mask_model.pt")

# 2. Định nghĩa tên class và màu sắc
classNames = ["wrong mask", "mask", "no mask"] 

# Màu sắc cho khung (chuẩn BGR của OpenCV)
colors = [
    (0, 255, 255),   # Vàng cho "Đeo sai" (wrong mask)
    (0, 255, 0),     # Xanh lá cho "Đeo đúng" (mask)
    (0, 0, 255)      # Đỏ cho "Không đeo" (no mask)
]

# Cấu hình kiểm tra trạng thái
CONF_THRESHOLD = 0.80     # Độ chính xác từ 80% trở lên
STABLE_DURATION = 3.0    # Cần duy trì trong 3 giây
LOST_TOLERANCE = 1.0     # Cho phép mất dấu tạm thời tối đa 1.0 giây

last_notified_status = None
face_tracks = {}
next_custom_id = 1000

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

# 3. Mở kết nối Webcam máy tính (số 0 là camera mặc định)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Không thể mở camera!")
        break

    current_time = time.time()
    results = model.track(img, persist=True, verbose=False)

    detected_faces = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = classNames[cls] if cls < len(classNames) else "Unknown"
            color = colors[cls] if cls < len(colors) else (255, 255, 255)

            track_id = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else None

            detected_faces.append({
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "cls": cls,
                "name": name,
                "color": color,
                "track_id": track_id
            })

    # Xử lý gán ID cho từng khuôn mặt (fallback IoU khi tracker chưa gán ID)
    for det in detected_faces:
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

        # Vẽ khung hình chữ nhật quanh khuôn mặt nếu độ tin cậy > 50%
        if det["conf"] > 0.5:
            cv2.rectangle(img, (x1, y1), (x2, y2), det["color"], 2)
            label = f"ID:{track_id} {det['name']} {int(det['conf'] * 100)}%"
            cv2.putText(img, label, (x1, max(30, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, det["color"], 2)

        # Logic theo dõi và thông báo (cả đổi khuôn mặt & đổi trạng thái) với conf >= 80%
        if det["conf"] >= CONF_THRESHOLD:
            name = det["name"]
            conf = det["conf"]
            if track_id not in face_tracks:
                # Phát hiện khuôn mặt mới
                face_tracks[track_id] = {
                    "status": name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "start_time": current_time,
                    "last_seen": current_time,
                    "last_notified_status": None,
                    "notified": False
                }
            else:
                track_info = face_tracks[track_id]
                track_info["box"] = (x1, y1, x2, y2)
                track_info["last_seen"] = current_time
                track_info["confidence"] = conf

                if name == track_info["status"]:
                    # Đang duy trì trạng thái
                    duration = current_time - track_info["start_time"]
                    if duration >= STABLE_DURATION:
                        # Điều kiện thông báo: Đổi khuôn mặt (chưa thông báo) HOẶC Đổi trạng thái
                        should_notify = (not track_info["notified"]) or (track_info["status"] != track_info["last_notified_status"])
                        if should_notify:
                            notify_reason = "Đổi khuôn mặt" if not track_info["notified"] else "Đổi trạng thái"
                            track_info["notified"] = True
                            track_info["last_notified_status"] = track_info["status"]
                            last_notified_status = track_info["status"]

                            timestamp = time.strftime("%H:%M:%S")
                            print(f"[{timestamp}] [THÔNG BÁO] [{notify_reason}] ID:{track_id} -> {track_info['status']} ({conf * 100:.0f}%) duy trì >= 3s")
                else:
                    # Khuôn mặt thay đổi trạng thái -> Reset đếm 3 giây
                    track_info["status"] = name
                    track_info["start_time"] = current_time

    # Xóa các khuôn mặt mất dấu quá LOST_TOLERANCE
    lost_ids = [fid for fid, f_data in face_tracks.items() if current_time - f_data["last_seen"] > LOST_TOLERANCE]
    for fid in lost_ids:
        del face_tracks[fid]

    # Hiển thị thông tin trạng thái và bộ đếm thời gian trên khung hình
    overlay_text = f"Status: {last_notified_status if last_notified_status else 'None'}"
    pending_items = []
    for fid, f_data in face_tracks.items():
        if (not f_data["notified"]) or (f_data["status"] != f_data["last_notified_status"]):
            elapsed = min(STABLE_DURATION, current_time - f_data["start_time"])
            pending_items.append(f"ID:{fid} {f_data['status']} ({elapsed:.1f}s/{STABLE_DURATION:.0f}s)")

    if pending_items:
        overlay_text += f" | Pending: {', '.join(pending_items)}"

    (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (10, 10), (10 + tw + 10, 10 + th + 14), (0, 0, 0), -1)
    text_color = (0, 255, 255) if pending_items else (0, 255, 0)
    cv2.putText(img, overlay_text, (15, 10 + th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Hiển thị cửa sổ Video
    cv2.imshow("Face Mask Detection", img)

    # Nhấn phím 'q' trên bàn phím để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và tắt camera
cap.release()
cv2.destroyAllWindows()
