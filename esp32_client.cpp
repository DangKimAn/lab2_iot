#include <Arduino.h>
#include "esp_camera.h"
#include "img_converters.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h> // BẠN CẦN CÀI THƯ VIỆN "ArduinoJson" TRONG ARDUINO IDE!
#include "board_config.h"

// ============================================================
// WiFi & Server Configuration
// ============================================================
const char *ssid = "Trai to doc";
const char *password = "giongtenwifi";

// Sửa IP này thành IP của EC2 AWS
const String SERVER_URL = "http://127.0.0.1:5000/api/v2/detect"; 
const String UPDATE_URL = "http://127.0.0.1:5000/api/v2/update_latency";

#define JPEG_QUALITY   70

// ============================================================
// SETUP
// ============================================================
void setup()
{
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  delay(1000);

  Serial.println("\n================================");
  Serial.println(" ESP32-CAM IoT EDGE CLIENT");
  Serial.println("================================");

  // 1. Kiểm tra PSRAM
  if (!psramFound()) {
    Serial.println("[PSRAM] ERROR: PSRAM not found!");
    while (true) { delay(1000); }
  }

  // 2. Cấu hình Camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;

  // GC2145 không hỗ trợ JPEG native, phải dùng RGB565 rồi encode mềm
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size   = FRAMESIZE_QVGA;
  config.jpeg_quality = JPEG_QUALITY;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAMERA] ERROR: init failed: 0x%x\n", err);
    while (true) { delay(1000); }
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_hmirror(s, 1);
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_framesize(s, FRAMESIZE_QVGA);
  }

  // 3. Kết nối WiFi
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  Serial.printf("[WIFI] Connecting to %s...\n", ssid);
  WiFi.begin(ssid, password);

  uint32_t wifi_start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (millis() - wifi_start > 30000) {
      Serial.println("\n[WIFI] ERROR: connection timeout");
      ESP.restart(); // Tự khởi động lại nếu không vào được mạng
    }
  }

  Serial.println("\n[WIFI] Connected! IP: ");
  Serial.println(WiFi.localIP());
}

// ============================================================
// LOOP - GỬI ẢNH LÊN CLOUD VÀ ĐO ĐỘ TRỄ
// ============================================================
void loop()
{
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi Disconnected. Reconnecting...");
    WiFi.reconnect();
    delay(5000);
    return;
  }

  // 1. Chụp ảnh
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[CAMERA] Capture failed");
    return;
  }

  // 2. Chuyển RGB565 sang JPEG
  uint8_t *jpg_buf = NULL;
  size_t jpg_len = 0;
  bool converted = frame2jpg(fb, JPEG_QUALITY, &jpg_buf, &jpg_len);
  esp_camera_fb_return(fb); // Trả lại buffer camera ngay cho trống RAM

  if (converted && jpg_buf && jpg_len > 0) {
    
    // =================================================
    // BẮT ĐẦU BẤM GIỜ
    // =================================================
    uint32_t start_time = millis();

    HTTPClient http;
    http.begin(SERVER_URL);
    
    // 3. Xây dựng gói tin multipart/form-data
    String boundary = "ESP32Boundary123456789";
    String head = "--" + boundary + "\r\n";
    head += "Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n";
    head += "Content-Type: image/jpeg\r\n\r\n";
    String tail = "\r\n--" + boundary + "--\r\n";
    
    size_t totalLen = head.length() + jpg_len + tail.length();
    
    // Cấp phát vùng nhớ lớn trên PSRAM để gộp gói tin
    uint8_t *payload = (uint8_t *)ps_malloc(totalLen);
    if (payload == NULL) {
      payload = (uint8_t *)malloc(totalLen);
    }
    
    if (payload != NULL) {
      // Gộp Header, Data ảnh, và Tail
      memcpy(payload, head.c_str(), head.length());
      memcpy(payload + head.length(), jpg_buf, jpg_len);
      memcpy(payload + head.length() + jpg_len, tail.c_str(), tail.length());
      
      http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
      
      // 4. Bắn lệnh POST
      int httpResponseCode = http.POST(payload, totalLen);
      
      // =================================================
      // KẾT THÚC BẤM GIỜ
      // =================================================
      uint32_t end_time = millis();
      float e2e_latency_ms = (float)(end_time - start_time);
      
      if (httpResponseCode == 200) {
        String responseBody = http.getString();
        
        // 5. Bóc tách JSON
        JsonDocument doc; // Yêu cầu ArduinoJson v7
        DeserializationError error = deserializeJson(doc, responseBody);
        
        if (!error) {
          float inference_time_ms = doc["inference_time_ms"];
          int detections = doc["detections"];
          float network_latency_ms = e2e_latency_ms - inference_time_ms;
          
          Serial.printf("[Latency] Toàn trình: %.1fms | AI: %.1fms | Mạng: %.1fms | Khuôn mặt: %d\n", 
                        e2e_latency_ms, inference_time_ms, network_latency_ms, detections);
                        
          // 6. KIỂM TRA 2-STEP LOGGING
          if (doc.containsKey("log_id") && !doc["log_id"].isNull()) {
            int log_id = doc["log_id"];
            Serial.printf("  -> Có sự kiện mới (log_id: %d). Đang cập nhật DB...\n", log_id);
            
            HTTPClient updateHttp;
            updateHttp.begin(UPDATE_URL);
            updateHttp.addHeader("Content-Type", "application/json");
            
            // Tạo chuỗi JSON gửi đi
            String jsonPayload = "{\"log_id\":" + String(log_id) + 
                                 ",\"network_latency_ms\":" + String(network_latency_ms) + 
                                 ",\"e2e_latency_ms\":" + String(e2e_latency_ms) + "}";
                                 
            int updateRes = updateHttp.POST(jsonPayload);
            if (updateRes == 200) {
              Serial.println("  -> Cập nhật Database thành công!");
            } else {
              Serial.printf("  -> Lỗi cập nhật Database: %d\n", updateRes);
            }
            updateHttp.end();
          }
        }
      } else {
        Serial.printf("[HTTP] Error code: %d\n", httpResponseCode);
      }
      
      free(payload);
    }
    http.end();
    free(jpg_buf);
  } else {
    Serial.println("[CAMERA] JPEG conversion failed");
  }
  
  // Tránh spam server, delay 100ms tương đương 10 FPS
  delay(100);
}
