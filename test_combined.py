"""
Filter v2: Bộ Lọc Chất Lượng Ảnh Label
======================================

Chức năng:
  - Lọc ảnh theo 4 tiêu chí: size, contrast, sharpness, brightness
  - [Option] Chuẩn hóa ảnh FIXABLE về mức target
  - [Option] Phát hiện QR Code
  - [Option] Nhận dạng chữ (OCR)

Usage:
  # Chỉ lọc (mặc định)
  python test_combined.py
  
  # Lọc + Chuẩn hóa ảnh FIXABLE
  python test_combined.py --normalize
  
  # Lọc + QR Detection
  python test_combined.py --qr
  
  # Lọc + Chuẩn hóa + QR + OCR (đầy đủ)
  python test_combined.py --normalize --qr --ocr

Output:
  Output/
  ├── 1_discard/     - Ảnh bị loại
  ├── 2_fixable/     - Ảnh cần xử lý (hoặc đã xử lý nếu --normalize)
  └── 3_good/        - Ảnh tốt
"""

import os
import sys
import shutil
import json
import argparse
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Tuple, Optional
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# ENUM & DATACLASS
# ==============================================================================
class QualityOutcome(Enum):
    GOOD = "good"
    FIXABLE = "fixable"
    DISCARD = "discard"


@dataclass
class Metrics:
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float


# ==============================================================================
# NGƯỠNG LỌC (S3b)
# ==============================================================================
FILTER_THRESHOLDS = {
    "size": {
        "good": {"minWidth": 300, "minHeight": 200},
        "fixable": {"minWidth": 200, "minHeight": 150}
    },
    "contrast": {
        "good": 55,       # >= 55 là tốt, không cần xử lý
        "fixable": 50     # >= 50 mới lấy, < 50 bị loại
    },
    "sharpness": {
        "good": 500,      # >= 500 là sắc nét
        "fixable": 200    # 200-499 cần normalize
    },
    "brightness": {
        "good": {"min": 80, "max": 220},
        "fixable": {"min": 60, "max": 240}  # Giữ nguyên như ban đầu
    }
}

# ==============================================================================
# MỨC CHUẨN TARGET (S4)
# ==============================================================================
NORMALIZE_TARGET = {
    "size": (300, 200),           # width, height
    "brightness": 150,            # mean pixel value
    "contrast": 60,               # std deviation
    "sharpness": 600,             # Laplacian variance
}


# ==============================================================================
# HÀM TÍNH METRICS
# ==============================================================================
def calculate_metrics(image: np.ndarray) -> Metrics:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    
    return Metrics(
        width=width,
        height=height,
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        sharpness=round(sharpness, 2)
    )


# ==============================================================================
# BỘ LỌC (S3b): Phân loại GOOD / FIXABLE / DISCARD
# ==============================================================================
def filter_image(metrics: Metrics) -> Tuple[QualityOutcome, Optional[str], dict]:
    """
    Phân loại ảnh thành 3 mức.
    
    Returns:
        (outcome, discard_reason, needs_fix)
        needs_fix = {"size": bool, "brightness": bool, "contrast": bool, "sharpness": bool}
    """
    t = FILTER_THRESHOLDS
    needs_fix = {"size": False, "brightness": False, "contrast": False, "sharpness": False}
    
    # Check SIZE
    if metrics.width < t["size"]["fixable"]["minWidth"] or \
       metrics.height < t["size"]["fixable"]["minHeight"]:
        return QualityOutcome.DISCARD, f"Size quá nhỏ ({metrics.width}x{metrics.height})", needs_fix
    elif metrics.width < t["size"]["good"]["minWidth"] or \
         metrics.height < t["size"]["good"]["minHeight"]:
        needs_fix["size"] = True
    
    # Check CONTRAST
    if metrics.contrast < t["contrast"]["fixable"]:
        return QualityOutcome.DISCARD, f"Contrast quá thấp ({metrics.contrast})", needs_fix
    elif metrics.contrast < t["contrast"]["good"]:
        needs_fix["contrast"] = True
    
    # Check SHARPNESS
    if metrics.sharpness < t["sharpness"]["fixable"]:
        return QualityOutcome.DISCARD, f"Ảnh quá mờ ({metrics.sharpness})", needs_fix
    elif metrics.sharpness < t["sharpness"]["good"]:
        needs_fix["sharpness"] = True
    
    # Check BRIGHTNESS
    if metrics.brightness < t["brightness"]["fixable"]["min"]:
        return QualityOutcome.DISCARD, f"Ảnh quá tối ({metrics.brightness})", needs_fix
    elif metrics.brightness > t["brightness"]["fixable"]["max"]:
        return QualityOutcome.DISCARD, f"Ảnh quá sáng ({metrics.brightness})", needs_fix
    elif metrics.brightness < t["brightness"]["good"]["min"] or \
         metrics.brightness > t["brightness"]["good"]["max"]:
        needs_fix["brightness"] = True
    
    # Quyết định
    if any(needs_fix.values()):
        return QualityOutcome.FIXABLE, None, needs_fix
    else:
        return QualityOutcome.GOOD, None, needs_fix


# ==============================================================================
# BỘ CHUẨN HÓA (S4): Normalize về target - CHỈ XỬ LÝ NHỮNG GÌ CẦN FIX
# ==============================================================================
def normalize_size(image: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    target_w, target_h = target
    
    if w == target_w and h == target_h:
        return image.copy()
    
    if w < target_w or h < target_h:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA
    
    return cv2.resize(image, target, interpolation=interpolation)


def normalize_brightness(image: np.ndarray, target: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current = float(np.mean(gray))
    
    if abs(current - target) < 5:
        return image.copy()
    
    adjustment = target - current
    
    if abs(adjustment) > 30:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip_limit = min(abs(adjustment) / 10, 4.0)
        clip_limit = max(clip_limit, 2.0)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        after_mean = float(np.mean(l))
        remaining = target - after_mean
        if abs(remaining) > 10:
            l = cv2.add(l, int(remaining * 0.5))
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return cv2.convertScaleAbs(image, alpha=1.0, beta=adjustment)


def normalize_contrast(image: np.ndarray, target: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    current = float(np.std(gray))
    
    if abs(current - target) < 5 or current < 1:
        return image.copy()
    
    scale = np.clip(target / current, 0.5, 2.5)
    
    result = image.astype(np.float32)
    for c in range(3):
        mean = np.mean(result[:, :, c])
        result[:, :, c] = (result[:, :, c] - mean) * scale + mean
    
    return np.clip(result, 0, 255).astype(np.uint8)


def normalize_sharpness(image: np.ndarray, target: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    current = float(laplacian.var())
    
    if current >= target or current < 10:
        return image.copy()
    
    ratio = target / current
    amount = np.clip((ratio - 1) * 0.8, 0.5, 3.0)
    
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def normalize_image(image: np.ndarray, needs_fix: dict) -> np.ndarray:
    """
    Chỉ normalize những gì cần fix.
    """
    result = image.copy()
    
    if needs_fix["size"]:
        result = normalize_size(result, NORMALIZE_TARGET["size"])
    
    if needs_fix["brightness"]:
        result = normalize_brightness(result, NORMALIZE_TARGET["brightness"])
    
    if needs_fix["contrast"]:
        result = normalize_contrast(result, NORMALIZE_TARGET["contrast"])
    
    if needs_fix["sharpness"]:
        result = normalize_sharpness(result, NORMALIZE_TARGET["sharpness"])
    
    return result


# ==============================================================================
# QR DETECTION (S5)
# ==============================================================================
def detect_qr(image: np.ndarray) -> Tuple[bool, Optional[str]]:
    try:
        import zxingcpp
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = zxingcpp.read_barcodes(gray)
        
        if barcodes:
            for bc in barcodes:
                if bc.valid:
                    return True, bc.text
        return False, None
    except Exception as e:
        return False, str(e)


# ==============================================================================
# OCR DETECTION (S7) - Singleton để tránh load model nhiều lần
# ==============================================================================
_ocr_engine = None

def get_ocr_engine():
    """Lazy load PaddleOCR engine."""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        import logging
        # Suppress PaddleOCR logs
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        _ocr_engine = PaddleOCR(
            lang='en',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        print("   PaddleOCR engine loaded.")
    return _ocr_engine


def detect_ocr(image: np.ndarray) -> Tuple[bool, list]:
    """
    Detect text using PaddleOCR.
    
    Returns:
        (success, texts): (bool, list of detected texts)
    """
    try:
        ocr = get_ocr_engine()
        result = ocr.predict(image)
        
        texts = []
        if result:
            for res in result:
                rec_texts = res.get('rec_texts', [])
                rec_scores = res.get('rec_scores', [])
                for i, text in enumerate(rec_texts):
                    score = rec_scores[i] if i < len(rec_scores) else 0
                    if score > 0.5:  # Chỉ lấy text confidence > 0.5
                        texts.append({"text": text, "confidence": round(score, 3)})
        
        return len(texts) > 0, texts
    except Exception as e:
        return False, [{"error": str(e)}]


def save_image_info(folder: str, filename: str, info: dict):
    """Save JSON info file alongside image."""
    json_filename = os.path.splitext(filename)[0] + "_info.json"
    json_path = os.path.join(folder, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================
def process_images(input_dir: str, output_dir: str, 
                   enable_normalize: bool = False,
                   enable_qr: bool = False,
                   enable_ocr: bool = False,
                   max_images: int = None):
    """
    Xử lý ảnh với các option.
    
    Args:
        input_dir: Thư mục chứa ảnh đầu vào
        output_dir: Thư mục lưu kết quả
        enable_normalize: Bật chuẩn hóa ảnh FIXABLE
        enable_qr: Bật phát hiện QR Code
        enable_ocr: Bật nhận dạng chữ (OCR)
        max_images: Giới hạn số ảnh xử lý (None = tất cả)
    """
    # Tạo thư mục output
    if enable_qr:
        # Nếu bật QR, chia thành detected/not_detected
        dirs = {
            "discard": os.path.join(output_dir, "1_discard"),
            "fixable_detected": os.path.join(output_dir, "2_fixable", "detected"),
            "fixable_not_detected": os.path.join(output_dir, "2_fixable", "not_detected"),
            "good_detected": os.path.join(output_dir, "3_good", "detected"),
            "good_not_detected": os.path.join(output_dir, "3_good", "not_detected"),
        }
    else:
        # Chỉ lọc, không cần chia detected/not_detected
        dirs = {
            "discard": os.path.join(output_dir, "1_discard"),
            "fixable": os.path.join(output_dir, "2_fixable"),
            "good": os.path.join(output_dir, "3_good"),
        }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if max_images:
        image_files = image_files[:max_images]
    
    # Header
    print("=" * 70)
    print("🔍 FILTER v2: BỘ LỌC CHẤT LƯỢNG ẢNH")
    print("=" * 70)
    print(f"📂 Input:  {input_dir}")
    print(f"📂 Output: {output_dir}")
    print(f"📷 Images: {len(image_files)}")
    print(f"")
    print(f"⚙️  Options:")
    print(f"   Normalize: {'✅ ON' if enable_normalize else '❌ OFF'}")
    print(f"   QR Detect: {'✅ ON' if enable_qr else '❌ OFF'}")
    print(f"   OCR:       {'✅ ON' if enable_ocr else '❌ OFF'}")
    print("=" * 70)
    
    stats = {
        "discard": 0,
        "good": 0,
        "fixable": 0,
        # Chi tiết khi bật QR
        "good_detected": 0,
        "good_not_detected": 0,
        "fixable_detected": 0,
        "fixable_not_detected": 0,
        # OCR stats
        "ocr_success": 0,
        "ocr_fail": 0,
    }
    
    results = []
    
    for i, filename in enumerate(image_files, 1):
        filepath = os.path.join(input_dir, filename)
        image = cv2.imread(filepath)
        
        if image is None:
            continue
        
        metrics = calculate_metrics(image)
        outcome, discard_reason, needs_fix = filter_image(metrics)
        
        # Khởi tạo biến
        qr_ok = None
        qr_text = None
        ocr_ok = None
        ocr_texts = []
        final_image = image
        fixes = [k for k, v in needs_fix.items() if v]
        
        if outcome == QualityOutcome.DISCARD:
            # Bỏ ảnh
            dest_folder = dirs["discard"]
            shutil.copy(filepath, os.path.join(dest_folder, filename))
            stats["discard"] += 1
            status = f"❌ DISCARD: {discard_reason}"
            
            # Lưu info
            save_image_info(dest_folder, filename, {
                "filename": filename,
                "outcome": "discard",
                "reason": discard_reason,
                "metrics": asdict(metrics),
                "thresholds": FILTER_THRESHOLDS
            })
            
        elif outcome == QualityOutcome.GOOD:
            stats["good"] += 1
            
            # QR Detection (nếu bật)
            if enable_qr:
                qr_ok, qr_text = detect_qr(image)
            
            # OCR (nếu bật)
            if enable_ocr:
                ocr_ok, ocr_texts = detect_ocr(image)
                if ocr_ok:
                    stats["ocr_success"] += 1
                else:
                    stats["ocr_fail"] += 1
            
            # Lưu ảnh
            if enable_qr:
                if qr_ok:
                    dest_folder = dirs["good_detected"]
                    stats["good_detected"] += 1
                    status = f"✅ GOOD → QR ✓"
                else:
                    dest_folder = dirs["good_not_detected"]
                    stats["good_not_detected"] += 1
                    status = f"✅ GOOD → QR ✗"
                if enable_ocr:
                    status += f" | OCR {'✓' if ocr_ok else '✗'}"
            else:
                dest_folder = dirs["good"]
                status = f"✅ GOOD"
                if enable_ocr:
                    status += f" | OCR {'✓' if ocr_ok else '✗'}"
            
            shutil.copy(filepath, os.path.join(dest_folder, filename))
            
            # Lưu info
            info = {
                "filename": filename,
                "outcome": "good",
                "metrics": asdict(metrics),
                "processing": "none"
            }
            if enable_qr:
                info["qr_detected"] = qr_ok
                info["qr_text"] = qr_text
            if enable_ocr:
                info["ocr_detected"] = ocr_ok
                info["ocr_texts"] = ocr_texts
            save_image_info(dest_folder, filename, info)
                
        else:  # FIXABLE
            stats["fixable"] += 1
            before_metrics = metrics
            
            # Normalize (nếu bật)
            if enable_normalize:
                final_image = normalize_image(image, needs_fix)
                after_metrics = calculate_metrics(final_image)
            else:
                final_image = image
                after_metrics = metrics
            
            # QR Detection (nếu bật)
            if enable_qr:
                qr_ok, qr_text = detect_qr(final_image)
            
            # OCR (nếu bật)
            if enable_ocr:
                ocr_ok, ocr_texts = detect_ocr(final_image)
                if ocr_ok:
                    stats["ocr_success"] += 1
                else:
                    stats["ocr_fail"] += 1
            
            # Lưu ảnh
            if enable_qr:
                if qr_ok:
                    dest_folder = dirs["fixable_detected"]
                    stats["fixable_detected"] += 1
                    status = f"🔧 FIXABLE ({', '.join(fixes)}) → QR ✓"
                else:
                    dest_folder = dirs["fixable_not_detected"]
                    stats["fixable_not_detected"] += 1
                    status = f"🔧 FIXABLE ({', '.join(fixes)}) → QR ✗"
                if enable_ocr:
                    status += f" | OCR {'✓' if ocr_ok else '✗'}"
            else:
                dest_folder = dirs["fixable"]
                status = f"🔧 FIXABLE ({', '.join(fixes)})"
                if enable_normalize:
                    status += " → Normalized"
                if enable_ocr:
                    status += f" | OCR {'✓' if ocr_ok else '✗'}"
            
            if enable_normalize:
                cv2.imwrite(os.path.join(dest_folder, filename), final_image)
            else:
                shutil.copy(filepath, os.path.join(dest_folder, filename))
            
            # Lưu info
            info = {
                "filename": filename,
                "outcome": "fixable",
                "needs_fix": fixes,
                "before_metrics": asdict(before_metrics),
                "processing": "normalized" if enable_normalize else "none"
            }
            if enable_normalize:
                info["after_metrics"] = asdict(after_metrics)
            if enable_qr:
                info["qr_detected"] = qr_ok
                info["qr_text"] = qr_text
            if enable_ocr:
                info["ocr_detected"] = ocr_ok
                info["ocr_texts"] = ocr_texts
            save_image_info(dest_folder, filename, info)
        
        # Log
        if i <= 15 or i % 10 == 0:
            print(f"[{i:3d}] {status}")
        
        results.append({
            "filename": filename,
            "outcome": outcome.value,
            "metrics": asdict(metrics),
            "needs_fix": needs_fix,
            "discard_reason": discard_reason,
            "qr_detected": qr_ok
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ")
    print("=" * 70)
    print(f"\n📋 Phân loại:")
    print(f"  ❌ DISCARD:  {stats['discard']:3d} ảnh (bỏ)")
    print(f"  ✅ GOOD:     {stats['good']:3d} ảnh (không cần xử lý)")
    print(f"  🔧 FIXABLE:  {stats['fixable']:3d} ảnh", end="")
    if enable_normalize:
        print(" (đã normalize)")
    else:
        print(" (cần xử lý)")
    
    if enable_qr:
        print(f"\n📍 QR Detection:")
        print(f"  GOOD → Detected:         {stats['good_detected']:3d}")
        print(f"  GOOD → Not detected:     {stats['good_not_detected']:3d}")
        print(f"  FIXABLE → Detected:      {stats['fixable_detected']:3d}")
        print(f"  FIXABLE → Not detected:  {stats['fixable_not_detected']:3d}")
        
        total_processed = stats['good'] + stats['fixable']
        total_detected = stats["good_detected"] + stats["fixable_detected"]
        if total_processed > 0:
            print(f"\n📈 Tỉ lệ QR Detected: {total_detected}/{total_processed} ({total_detected/total_processed*100:.1f}%)")
    
    if enable_ocr:
        total_ocr = stats['ocr_success'] + stats['ocr_fail']
        if total_ocr > 0:
            print(f"📈 Tỉ lệ OCR Success: {stats['ocr_success']}/{total_ocr} ({stats['ocr_success']/total_ocr*100:.1f}%)")
    
    print("=" * 70)
    
    # Save
    summary = {
        "timestamp": datetime.now().isoformat(),
        "options": {
            "normalize": enable_normalize,
            "qr": enable_qr,
            "ocr": enable_ocr
        },
        "filter_thresholds": FILTER_THRESHOLDS,
        "normalize_target": NORMALIZE_TARGET if enable_normalize else None,
        "stats": stats,
        "results": results
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter v2: Bộ lọc chất lượng ảnh label",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python test_combined.py                      # Chỉ lọc
  python test_combined.py --normalize          # Lọc + Chuẩn hóa
  python test_combined.py --qr                 # Lọc + QR Detection
  python test_combined.py --normalize --qr --ocr  # Đầy đủ
        """
    )
    
    parser.add_argument("-i", "--input", default="Input",
                        help="Thư mục chứa ảnh đầu vào (mặc định: Input)")
    parser.add_argument("-o", "--output", default="Output",
                        help="Thư mục lưu kết quả (mặc định: Output)")
    parser.add_argument("--normalize", action="store_true",
                        help="Bật chuẩn hóa ảnh FIXABLE")
    parser.add_argument("--qr", action="store_true",
                        help="Bật phát hiện QR Code")
    parser.add_argument("--ocr", action="store_true",
                        help="Bật nhận dạng chữ (OCR)")
    parser.add_argument("-n", "--max-images", type=int, default=None,
                        help="Giới hạn số ảnh xử lý")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input directory not found: {args.input}")
        sys.exit(1)
    
    process_images(
        input_dir=args.input,
        output_dir=args.output,
        enable_normalize=args.normalize,
        enable_qr=args.qr,
        enable_ocr=args.ocr,
        max_images=args.max_images
    )
