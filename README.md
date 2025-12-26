# 🔍 Filter v2 - Bộ Lọc & Chuẩn Hóa Ảnh Label

Công cụ xử lý ảnh label đã crop từ bước S3 (Preprocessing) của luồng chính, thực hiện:
- **Lọc chất lượng** ảnh (size, contrast, sharpness, brightness)
- **Chuẩn hóa** ảnh không đạt chuẩn
- **Phát hiện QR Code** và **OCR** trên ảnh

---

## 📋 Mục Lục

1. [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
2. [Cài Đặt](#-cài-đặt)
3. [Cách Sử Dụng](#-cách-sử-dụng)
4. [Cấu Trúc Đầu Vào/Đầu Ra](#-cấu-trúc-đầu-vàođầu-ra)
5. [Luồng Xử Lý](#-luồng-xử-lý)
6. [Ngưỡng Phân Loại](#-ngưỡng-phân-loại)

---

## 💻 Yêu Cầu Hệ Thống

- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB do sử dụng PaddleOCR)
- **OS**: Linux / Windows / macOS

---

## 🔧 Cài Đặt

### Bước 1: Clone Repository

```bash
git clone https://github.com/hongphuoc6104/Filter_v2.git
cd Filter_v2
```

### Bước 2: Tạo Virtual Environment (Khuyến nghị)

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Bước 3: Cài Đặt Thư Viện

```bash
pip install opencv-python numpy paddleocr zxing-cpp
```

**Chi tiết các thư viện:**

| Thư viện | Mục đích | Lệnh cài đặt |
|----------|----------|--------------|
| opencv-python | Xử lý ảnh | `pip install opencv-python` |
| numpy | Tính toán số học | `pip install numpy` |
| paddleocr | Nhận dạng chữ (OCR) | `pip install paddleocr` |
| zxing-cpp | Đọc QR Code | `pip install zxing-cpp` |

> ⚠️ **Lưu ý**: Lần đầu chạy, PaddleOCR sẽ tự động tải model (~100MB), cần kết nối internet.

---

## 🚀 Cách Sử Dụng

### Bước 1: Chuẩn Bị Ảnh Đầu Vào

Đặt các ảnh label đã crop (từ bước S3 Preprocessing của luồng chính) vào thư mục `Input/`:

```
Filter_v2/
├── Input/
│   ├── label_001.png
│   ├── label_002.png
│   ├── label_003.png
│   └── ...
```

### Bước 2: Chạy Script

**Chỉ lọc (mặc định):**
```bash
python test_combined.py
```

**Lọc + Chuẩn hóa ảnh FIXABLE:**
```bash
python test_combined.py --normalize
```

**Lọc + Phát hiện QR Code:**
```bash
python test_combined.py --qr
```

**Lọc + OCR:**
```bash
python test_combined.py --ocr
```

**Đầy đủ tất cả chức năng:**
```bash
python test_combined.py --normalize --qr --ocr
```

**Chỉ định thư mục tùy chỉnh:**
```bash
python test_combined.py -i my_images -o results --normalize --qr
```

**Xem tất cả options:**
```bash
python test_combined.py --help
```

### Bảng Các Options

| Option | Mô tả |
|--------|-------|
| `-i`, `--input` | Thư mục đầu vào (mặc định: `Input`) |
| `-o`, `--output` | Thư mục đầu ra (mặc định: `Output`) |
| `--normalize` | Bật chuẩn hóa ảnh FIXABLE |
| `--qr` | Bật phát hiện QR Code |
| `--ocr` | Bật nhận dạng chữ (OCR) |
| `-n`, `--max-images` | Giới hạn số ảnh xử lý |

### Bước 3: Xem Kết Quả

Kết quả được lưu trong thư mục `Output/`. Xem file `summary.json` để có thống kê tổng hợp.

---

## 📁 Cấu Trúc Đầu Vào/Đầu Ra

### Đầu Vào (Input)
```
Input/
├── image1.png      # Ảnh label đã crop từ S3
├── image2.png
└── ...
```

> 📌 **Lưu ý**: Chỉ hỗ trợ file `.png`

### Đầu Ra (Output)
```
Output/
├── 1_discard/              # ❌ Ảnh bị loại (quá kém chất lượng)
│   ├── image_kém.png
│   └── image_kém_info.json
│
├── 2_fixable/              # 🔧 Ảnh đã được chuẩn hóa
│   ├── detected/           # ✅ Chuẩn hóa + QR OK
│   │   ├── image.png
│   │   └── image_info.json
│   └── not_detected/       # ❌ Chuẩn hóa + QR không tìm thấy
│       ├── image.png
│       └── image_info.json
│
├── 3_good/                 # ✅ Ảnh tốt (không cần xử lý)
│   ├── detected/           # ✅ QR OK
│   │   ├── image.png
│   │   └── image_info.json
│   └── not_detected/       # ❌ QR không tìm thấy
│       ├── image.png
│       └── image_info.json
│
└── summary.json            # 📊 Thống kê tổng hợp
```

---

## 🔄 Luồng Xử Lý

```
Ảnh Label (từ S3)
       │
       ▼
┌──────────────────┐
│   BỘ LỌC (S3b)   │  ← Phân loại theo 4 tiêu chí
└──────────────────┘
       │
       ├─── GOOD ──────► Không xử lý ──► QR Detection + OCR
       │
       ├─── FIXABLE ───► Chuẩn hóa ────► QR Detection + OCR
       │
       └─── DISCARD ───► Bỏ (lưu vào 1_discard/)
```

---

## 📏 Ngưỡng Phân Loại

| Tiêu Chí | GOOD ✅ | FIXABLE 🔧 | DISCARD ❌ |
|----------|---------|------------|------------|
| **Kích thước** | ≥ 300×200 px | 200×150 - 300×200 px | < 200×150 px |
| **Độ tương phản** | ≥ 50 | 30 - 49 | < 30 |
| **Độ nét** | ≥ 500 | 200 - 499 | < 200 |
| **Độ sáng** | 80 - 220 | 60 - 240 | < 60 hoặc > 240 |

### Mức Chuẩn (Target) Khi Normalize

| Tiêu chí | Giá trị target |
|----------|----------------|
| Kích thước | 300 × 200 px |
| Độ sáng | 150 |
| Độ tương phản | 60 |
| Độ nét | 600 |

---

## 📊 Đọc Kết Quả

### File `summary.json`

```json
{
  "stats": {
    "discard": 5,
    "good_detected": 20,
    "good_not_detected": 3,
    "fixable_detected": 15,
    "fixable_not_detected": 7,
    "ocr_success": 38,
    "ocr_fail": 2
  }
}
```

### File `*_info.json` (mỗi ảnh)

Chứa thông tin chi tiết về:
- Metrics (width, height, brightness, contrast, sharpness)
- Kết quả QR detection
- Kết quả OCR
- Lý do bị loại (nếu có)

---

## 🐛 Xử Lý Lỗi Thường Gặp

### 1. Lỗi `ModuleNotFoundError: No module named 'xxx'`
```bash
pip install <tên_module>
```

### 2. Lỗi PaddleOCR không tải được model
- Kiểm tra kết nối internet
- Thử: `pip install paddlepaddle paddleocr --upgrade`

### 3. Lỗi `zxingcpp` trên Linux
```bash
sudo apt-get install libzxing-dev
pip install zxing-cpp
```

---

## 📝 License

MIT License

---

## 👤 Tác Giả

GitHub: [@hongphuoc6104](https://github.com/hongphuoc6104)
