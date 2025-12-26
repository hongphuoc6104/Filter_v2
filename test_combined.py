"""
Test Combined: Bộ Lọc + Bộ Chuẩn Hóa
=====================================

Flow:
  Ảnh → BỘ LỌC (S3b) 
           │
           ├── GOOD     → Không xử lý → S5 QR
           ├── FIXABLE  → BỘ CHUẨN HÓA (S4) → S5 QR  
           └── DISCARD  → Bỏ

Output:
  output/combined_test/
  ├── 1_discard/           - Ảnh bỏ (quá kém)
  ├── 2_fixable/
  │   ├── before/          - Ảnh trước normalize
  │   ├── after/           - Ảnh sau normalize
  │   ├── detected/        - Normalize + QR OK
  │   └── not_detected/    - Normalize + QR fail
  └── 3_good/
      ├── detected/        - Không cần xử lý + QR OK
      └── not_detected/    - Không cần xử lý + QR fail

Usage:
    cd YOLO11-Seg-Label-Detector
    source .venv/bin/activate
    python scripts/test_combined.py
"""

import os
import sys
import shutil
import json
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
        "good": 50,       # >= 50 là tốt, không cần xử lý
        "fixable": 30     # 30-49 cần normalize
    },
    "sharpness": {
        "good": 500,      # >= 500 là sắc nét
        "fixable": 200    # 200-499 cần normalize
    },
    "brightness": {
        "good": {"min": 80, "max": 220},
        "fixable": {"min": 60, "max": 240}
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
def process_images(input_dir: str, output_dir: str, max_images: int = None):
    # Tạo thư mục (giống smart_filter_test)
    dirs = {
        "discard": os.path.join(output_dir, "1_discard"),
        "fixable_detected": os.path.join(output_dir, "2_fixable", "detected"),
        "fixable_not_detected": os.path.join(output_dir, "2_fixable", "not_detected"),
        "good_detected": os.path.join(output_dir, "3_good", "detected"),
        "good_not_detected": os.path.join(output_dir, "3_good", "not_detected"),
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    if max_images:
        image_files = image_files[:max_images]
    
    print("=" * 70)
    print("🔧 TEST COMBINED: BỘ LỌC + BỘ CHUẨN HÓA")
    print("=" * 70)
    print("Flow:")
    print("  Ảnh → S3b LỌC → GOOD     → Không xử lý → S5 QR")
    print("                → FIXABLE  → S4 NORMALIZE → S5 QR")
    print("                → DISCARD  → Bỏ")
    print("=" * 70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(image_files)}")
    print("=" * 70)
    
    stats = {
        "discard": 0,
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
        
        if outcome == QualityOutcome.DISCARD:
            # Bỏ ảnh
            dest_folder = dirs["discard"]
            shutil.copy(filepath, os.path.join(dest_folder, filename))
            stats["discard"] += 1
            status = f"❌ DISCARD: {discard_reason}"
            qr_ok = False
            
            # Lưu info
            save_image_info(dest_folder, filename, {
                "filename": filename,
                "outcome": "discard",
                "reason": discard_reason,
                "metrics": asdict(metrics),
                "thresholds": FILTER_THRESHOLDS
            })
            
        elif outcome == QualityOutcome.GOOD:
            # Không cần xử lý, detect QR trực tiếp
            qr_ok, qr_text = detect_qr(image)
            ocr_ok, ocr_texts = detect_ocr(image)
            
            if ocr_ok:
                stats["ocr_success"] += 1
            else:
                stats["ocr_fail"] += 1
            
            if qr_ok:
                dest_folder = dirs["good_detected"]
                shutil.copy(filepath, os.path.join(dest_folder, filename))
                stats["good_detected"] += 1
                status = f"✅ GOOD → QR ✓ | OCR {'✓' if ocr_ok else '✗'}"
                
                # Lưu info
                save_image_info(dest_folder, filename, {
                    "filename": filename,
                    "outcome": "good",
                    "qr_detected": True,
                    "qr_text": qr_text,
                    "ocr_detected": ocr_ok,
                    "ocr_texts": ocr_texts,
                    "metrics": asdict(metrics),
                    "processing": "none"
                })
            else:
                dest_folder = dirs["good_not_detected"]
                shutil.copy(filepath, os.path.join(dest_folder, filename))
                stats["good_not_detected"] += 1
                status = f"✅ GOOD → QR ✗ | OCR {'✓' if ocr_ok else '✗'}"
                
                # Lưu info - lý do không detect được
                save_image_info(dest_folder, filename, {
                    "filename": filename,
                    "outcome": "good",
                    "qr_detected": False,
                    "qr_fail_reason": "QR code not found or unreadable in image",
                    "ocr_detected": ocr_ok,
                    "ocr_texts": ocr_texts,
                    "metrics": asdict(metrics),
                    "processing": "none",
                    "suggestion": "Check if QR code is present, visible, and not damaged"
                })
                
        else:  # FIXABLE
            # Tính metrics trước khi fix
            before_metrics = metrics
            
            # Normalize CHỈ những gì cần fix
            normalized = normalize_image(image, needs_fix)
            
            # Tính metrics sau khi fix
            after_metrics = calculate_metrics(normalized)
            
            # Detect QR + OCR
            qr_ok, qr_text = detect_qr(normalized)
            ocr_ok, ocr_texts = detect_ocr(normalized)
            
            if ocr_ok:
                stats["ocr_success"] += 1
            else:
                stats["ocr_fail"] += 1
            
            fixes = [k for k, v in needs_fix.items() if v]
            
            if qr_ok:
                dest_folder = dirs["fixable_detected"]
                cv2.imwrite(os.path.join(dest_folder, filename), normalized)
                stats["fixable_detected"] += 1
                status = f"🔧 FIXABLE ({', '.join(fixes)}) → QR ✓ | OCR {'✓' if ocr_ok else '✗'}"
                
                # Lưu info
                save_image_info(dest_folder, filename, {
                    "filename": filename,
                    "outcome": "fixable",
                    "qr_detected": True,
                    "qr_text": qr_text,
                    "ocr_detected": ocr_ok,
                    "ocr_texts": ocr_texts,
                    "fixes_applied": fixes,
                    "before_metrics": asdict(before_metrics),
                    "after_metrics": asdict(after_metrics),
                    "processing": "normalized"
                })
            else:
                dest_folder = dirs["fixable_not_detected"]
                cv2.imwrite(os.path.join(dest_folder, filename), normalized)
                stats["fixable_not_detected"] += 1
                status = f"🔧 FIXABLE ({', '.join(fixes)}) → QR ✗ | OCR {'✓' if ocr_ok else '✗'}"
                
                # Lưu info - lý do không detect được
                save_image_info(dest_folder, filename, {
                    "filename": filename,
                    "outcome": "fixable",
                    "qr_detected": False,
                    "qr_fail_reason": "QR code not found after normalization",
                    "ocr_detected": ocr_ok,
                    "ocr_texts": ocr_texts,
                    "fixes_applied": fixes,
                    "before_metrics": asdict(before_metrics),
                    "after_metrics": asdict(after_metrics),
                    "processing": "normalized",
                    "possible_issues": [
                        "QR code may be damaged or partially visible",
                        "Image quality still insufficient after enhancement",
                        "QR code position may be outside detection area"
                    ]
                })
        
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
    total_good = stats["good_detected"] + stats["good_not_detected"]
    total_fixable = stats["fixable_detected"] + stats["fixable_not_detected"]
    total_processed = total_good + total_fixable
    total_detected = stats["good_detected"] + stats["fixable_detected"]
    
    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ")
    print("=" * 70)
    print(f"\n📋 Phân loại:")
    print(f"  ❌ DISCARD:  {stats['discard']:3d} ảnh (bỏ)")
    print(f"  ✅ GOOD:     {total_good:3d} ảnh (không cần xử lý)")
    print(f"  🔧 FIXABLE:  {total_fixable:3d} ảnh (đã normalize)")
    
    print(f"\n📍 QR Detection:")
    print(f"  GOOD → Detected:         {stats['good_detected']:3d}")
    print(f"  GOOD → Not detected:     {stats['good_not_detected']:3d}")
    print(f"  FIXABLE → Detected:      {stats['fixable_detected']:3d}")
    print(f"  FIXABLE → Not detected:  {stats['fixable_not_detected']:3d}")
    
    if total_processed > 0:
        print(f"\n📈 Tổng:")
        print(f"  QR Detected:  {total_detected}/{total_processed} ({total_detected/total_processed*100:.1f}%)")
        print(f"  OCR Success:  {stats['ocr_success']}/{total_processed} ({stats['ocr_success']/total_processed*100:.1f}%)")
    print("=" * 70)
    
    # Save
    summary = {
        "timestamp": datetime.now().isoformat(),
        "filter_thresholds": FILTER_THRESHOLDS,
        "normalize_target": NORMALIZE_TARGET,
        "stats": stats,
        "results": results
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    input_dir = "output/debug/s3_preprocessing"
    output_dir = "output/combined_test"
    
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    process_images(input_dir, output_dir)
