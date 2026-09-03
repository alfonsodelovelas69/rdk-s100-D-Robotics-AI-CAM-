
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RDK S100: YOLOE11 HBM vehicle detection + strict Ukrainian plate OCR."""

import argparse
import inspect
import os
import re
import shutil
import signal
import sys
import threading
import time
from collections import Counter
from datetime import datetime

import cv2
import hbm_runtime
import numpy as np
import pytesseract
from rapidfuzz import fuzz

# These are the same S100 helper modules used by the official YOLOE11 sample.
sys.path.append("/app/pydev_demo")
import utils.preprocess_utils as pre_utils
import utils.postprocess_utils as post_utils


MODEL_PATH = "/opt/hobot/model/s100/basic/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm"
DEFAULT_VIDEO = "/root/phone_video.mp4"
CROPS_DIR = "/root/crops"
DESKTOP_DIR = "/home/sunrise/Desktop/plate_crops"
LOG_FILE = "/root/plate_logs/scan.log"
NOTEBOOK_FILE = "/root/plates_notebook.txt"
DEFAULT_PIO_PIN = 17
DEFAULT_PIO_PULSE = 0.35

# Confirmed IDs from the prompt-free YOLOE11 model on this S100.
VEHICLE_IDS = {726, 4329}  # car, van


class PioTrigger:
    """Simple sysfs GPIO pulse for one PIO output on the RDK S100 board."""

    def __init__(self, pin=DEFAULT_PIO_PIN, active_high=True, sysfs_root="/sys/class/gpio", gpio_factory=None):
        self.pin = int(pin)
        self.active_high = bool(active_high)
        self.sysfs_root = sysfs_root
        self.gpio_factory = gpio_factory or self._sysfs_gpio
        self.gpio = self._create_gpio()
        self.lock = threading.Lock()
        self.last_state = 0

    def _create_gpio(self):
        factory = self.gpio_factory
        try:
            return factory(self.pin, self.active_high, self.sysfs_root)
        except TypeError:
            try:
                return factory(self.pin, self.active_high)
            except TypeError:
                raise

    @staticmethod
    def _sysfs_gpio(pin, active_high, sysfs_root):
        return _SysfsGPIO(pin, active_high, sysfs_root)

    def activate(self, duration=DEFAULT_PIO_PULSE):
        with self.lock:
            active_value = 1 if self.active_high else 0
            idle_value = 0 if self.active_high else 1
            self.last_state = active_value
            self.gpio.set_value(active_value)
            if duration > 0:
                time.sleep(duration)
            self.gpio.set_value(idle_value)
            self.last_state = active_value


class NoopTrigger:
    """Keep video processing alive when the requested PIO is unavailable."""

    def activate(self, duration=DEFAULT_PIO_PULSE):
        log("[WARN] PIO trigger skipped because the PIO device is unavailable")


class _SysfsGPIO:
    def __init__(self, pin, active_high=True, sysfs_root="/sys/class/gpio"):
        self.pin = int(pin)
        self.active_high = bool(active_high)
        self.sysfs_root = sysfs_root
        self.export_path = os.path.join(sysfs_root, "export")
        self.unexport_path = os.path.join(sysfs_root, "unexport")
        self.value_path = os.path.join(sysfs_root, "gpio%d" % self.pin, "value")
        self.direction_path = os.path.join(sysfs_root, "gpio%d" % self.pin, "direction")
        self._export()
        self._set_direction("out")
        self.set_value(0 if self.active_high else 1)

    def _export(self):
        if not os.path.exists(self.value_path):
            with open(self.export_path, "w", encoding="utf-8") as handle:
                handle.write(str(self.pin))
            for _ in range(50):
                if os.path.exists(self.value_path):
                    return
                time.sleep(0.02)
        if not os.path.exists(self.value_path):
            raise OSError("GPIO %d is not available in %s" % (self.pin, self.sysfs_root))

    def _set_direction(self, direction):
        with open(self.direction_path, "w", encoding="utf-8") as handle:
            handle.write(direction)

    def set_value(self, level):
        with open(self.value_path, "w", encoding="utf-8") as handle:
            handle.write("1" if int(level) else "0")


# Confirmed IDs from the prompt-free YOLOE11 model on this S100.

MODEL_SCORE = 0.40
MODEL_NMS = 0.70
MIN_VEHICLE_SCORE = 0.45
MIN_VEHICLE_WIDTH = 100
MIN_VEHICLE_HEIGHT = 60

MIN_PLATE_WIDTH = 55
MIN_PLATE_HEIGHT = 12
MIN_PLATE_RATIO = 2.2
MAX_PLATE_RATIO = 7.0
# Dirty or lightly scratched plates have weaker contours. Geometry and two
# temporal confirmations still protect us from saving wall fragments.
MIN_PLATE_SHARPNESS = 28.0

STOP = threading.Event()


def on_stop(_signum, _frame):
    """Signal handler; Event avoids a global assignment and SyntaxError risk."""
    STOP.set()


def log(message):
    line = "%s %s" % (datetime.now().strftime("%F %T"), message)
    print(line, flush=True)
    try:
        directory = os.path.dirname(LOG_FILE)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def clamp_box(box, width, height):
    x1, y1, x2, y2 = (int(value) for value in box)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def box_iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return float(intersection) / union if union > 0 else 0.0


def crop_valid(image):
    return image is not None and image.size > 0 and image.shape[0] >= 8 and image.shape[1] >= 20


def image_sharpness(image):
    if not crop_valid(image):
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class YoloE11Detector:
    """BBox-only part of the official S100 YOLOE11 segmentation postprocess."""

    def __init__(self, model_path, score_threshold, nms_threshold):
        log("[INFO] Loading YOLOE11:")
        log(model_path)
        self.model = hbm_runtime.HB_HBMRuntime(model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]
        self.input_h = self.input_shapes[self.input_names[0]][1]
        self.input_w = self.input_shapes[self.input_names[0]][2]
        self.nms_threshold = nms_threshold
        self.conf_threshold_raw = -np.log(1.0 / score_threshold - 1.0)
        self.weights_static = np.arange(16, dtype=np.float32)[None, None, :]

        try:
            self.model.set_scheduling_params(
                priority={self.model_name: 0},
                bpu_cores={self.model_name: [0]},
            )
        except Exception as exc:
            log("[WARN] Could not set BPU scheduling: %s" % exc)

        log("[INFO] Model name: %s" % self.model_name)
        log("[INFO] Input names: %s" % self.input_names)
        log("[INFO] Input size: %d x %d" % (self.input_w, self.input_h))
        log("[OK] YOLOE11 initialized on S100 BPU")

    def detect(self, image):
        original_h, original_w = image.shape[:2]

        # Keep these calls and their arguments aligned with the S100 sample.
        resized = pre_utils.resized_image(image, self.input_w, self.input_h, 1)
        image_y, image_uv = pre_utils.bgr_to_nv12_planes(resized)

        started = time.perf_counter()
        raw_outputs = self.model.run({
            self.model_name: {
                self.input_names[0]: image_y,
                self.input_names[1]: image_uv,
            }
        })
        bpu_ms = (time.perf_counter() - started) * 1000.0

        outputs = raw_outputs[self.model_name]
        fp32_outputs = post_utils.dequantize_outputs(outputs, self.output_quants)

        all_boxes = []
        all_scores = []
        all_ids = []
        for level, (stride, anchor_size) in enumerate(zip((8, 16, 32), (80, 40, 20))):
            class_key = self.output_names[3 * level]
            box_key = self.output_names[3 * level + 1]
            scores, class_ids, valid = post_utils.filter_classification(
                fp32_outputs[class_key], self.conf_threshold_raw
            )
            if len(scores) == 0:
                continue
            boxes = post_utils.decode_boxes(
                fp32_outputs[box_key], valid, anchor_size, stride, self.weights_static
            )
            if len(boxes) == 0:
                continue
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_ids.append(class_ids)

        if not all_boxes:
            return [], bpu_ms

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_ids, axis=0)
        keep = post_utils.NMS(boxes, scores, class_ids, self.nms_threshold)
        boxes = post_utils.scale_coords_back(
            boxes[keep], original_w, original_h, self.input_w, self.input_h, 1
        )

        detections = []
        for box, class_id, score in zip(boxes, class_ids[keep], scores[keep]):
            detections.append((
                clamp_box(box, original_w, original_h),
                int(class_id),
                float(score),
            ))
        return detections, bpu_ms


def vehicle_detections(detections):
    vehicles = []
    for box, class_id, score in detections:
        if class_id not in VEHICLE_IDS or score < MIN_VEHICLE_SCORE:
            continue
        x1, y1, x2, y2 = box
        if x2 - x1 < MIN_VEHICLE_WIDTH or y2 - y1 < MIN_VEHICLE_HEIGHT:
            continue
        vehicles.append((box, class_id, score))
    return vehicles


def find_plate_candidate(vehicle):
    """Return just one strong plate-shaped crop, never a whole vehicle crop."""
    if not crop_valid(vehicle):
        return None
    height, width = vehicle.shape[:2]
    if width < MIN_VEHICLE_WIDTH or height < MIN_VEHICLE_HEIGHT:
        return None

    roi_y = int(height * 0.42)
    roi = vehicle[roi_y:height, :]
    if not crop_valid(roi):
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blackhat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
    )
    binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 3)),
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, candidate_w, candidate_h = cv2.boundingRect(contour)
        if candidate_h <= 0:
            continue
        ratio = candidate_w / float(candidate_h)
        area = candidate_w * candidate_h
        if not (MIN_PLATE_RATIO <= ratio <= MAX_PLATE_RATIO):
            continue
        if candidate_w < MIN_PLATE_WIDTH or candidate_h < MIN_PLATE_HEIGHT or area < 700:
            continue
        if area > 0.10 * width * height:
            continue
        if x <= 2 or y <= 2 or x + candidate_w >= width - 2 or y + candidate_h >= roi.shape[0] - 2:
            continue

        center_x = (x + candidate_w / 2.0) / float(width)
        center_distance = abs(center_x - 0.5)
        if center_distance > 0.38:
            continue

        pad_x = max(4, int(candidate_w * 0.08))
        pad_y = max(3, int(candidate_h * 0.25))
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2 = min(width, x + candidate_w + pad_x)
        y2 = min(roi.shape[0], y + candidate_h + pad_y)
        crop = roi[y1:y2, x1:x2].copy()
        if not crop_valid(crop):
            continue

        crop_h, crop_w = crop.shape[:2]
        crop_ratio = crop_w / float(max(1, crop_h))
        if not (2.0 <= crop_ratio <= 8.0):
            continue
        sharpness = image_sharpness(crop)
        if sharpness < MIN_PLATE_SHARPNESS:
            continue
        brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
        if not (30.0 <= brightness <= 245.0):
            continue

        quality = area + sharpness * 7.0 - center_distance * area * 1.5
        candidates.append((quality, crop))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


# Unicode escapes prevent terminal encoding damage to Ukrainian letters.
CYRILLIC_TO_LATIN = str.maketrans({
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u0406": "I",
    "\u0407": "I", "\u041a": "K", "\u041c": "M", "\u041d": "H",
    "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T",
    "\u0425": "X",
})
DIGIT_TO_LETTER = str.maketrans("012458", "OIZESB")  # notably: 8 -> B
LETTER_TO_DIGIT = str.maketrans("OILZSB", "011258")


def normalize_plate(text):
    """Accept only a full 8-character Ukrainian-style plate: AA1234BB."""
    if not text:
        return ""
    cleaned = re.sub(
        r"[^A-Z0-9]", "", text.upper().translate(CYRILLIC_TO_LATIN)
    )
    if len(cleaned) < 8:
        return ""

    # OCR may prepend/append a stray character, so inspect every 8-char window.
    for start in range(len(cleaned) - 7):
        part = cleaned[start:start + 8]
        prefix = part[:2].translate(DIGIT_TO_LETTER)
        digits = part[2:6].translate(LETTER_TO_DIGIT)
        suffix = part[6:8].translate(DIGIT_TO_LETTER)
        if re.fullmatch(r"[A-Z]{2}", prefix) and re.fullmatch(r"[0-9]{4}", digits) and re.fullmatch(r"[A-Z]{2}", suffix):
            return prefix + digits + suffix
    return ""


def prepare_ocr_images(crop):
    if not crop_valid(crop):
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(8.0, max(3.0, 800.0 / max(1, width), 96.0 / max(1, height)))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Smooth dust speckles without erasing character boundaries.
    gray = cv2.bilateralFilter(gray, 7, 35, 35)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inverted = cv2.bitwise_not(otsu)
    # Reconnect character strokes interrupted by dust or small scratches.
    closed = cv2.morphologyEx(
        otsu,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    adaptive = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )

    def border(image):
        return cv2.copyMakeBorder(image, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)

    return [
        ("clahe", border(clahe)),
        ("otsu", border(otsu)),
        ("inverted", border(inverted)),
        ("closed", border(closed)),
        ("adaptive", border(adaptive)),
    ]


def run_ocr(crop, debug=False):
    images = prepare_ocr_images(crop)
    if not images:
        return []

    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    psm7 = "--oem 1 --psm 7 -c tessedit_char_whitelist=" + whitelist
    psm13 = "--oem 1 --psm 13 -c tessedit_char_whitelist=" + whitelist
    values = []
    raw = []

    def read(label, image, config):
        try:
            text = pytesseract.image_to_string(image, config=config, lang="eng")
        except Exception as exc:
            log("[WARN] OCR error: %s" % exc)
            return
        raw.append((label, text.strip()))
        value = normalize_plate(text)
        if value and value not in values:
            values.append(value)

    for label, image in images:
        read(label, image, psm7)
    if not values:
        for label, image in images[1:3]:
            read(label + "_psm13", image, psm13)

    if debug:
        height, width = crop.shape[:2]
        raw_summary = " | ".join("%s=%r" % item for item in raw)
        log("[OCR] crop=%dx%d parsed=%s raw=%s" % (width, height, values, raw_summary))
    return values


def ensure_notebook_file(path=NOTEBOOK_FILE):
    """Auto-create the notebook file when launched over SSH/PuTTY without a GUI."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if os.path.isfile(path):
        return path

    sample_lines = [
        "# One plate per line in format AA1234BB",
        "AA1234BB",
        "BC5678HX",
        "AK9901OP",
        "",
    ]
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(sample_lines))
        log("[INFO] Created notebook file: %s" % path)
    except OSError as exc:
        log("[WARN] Could not create notebook file %s: %s" % (path, exc))
    return path


def load_notebook(path=NOTEBOOK_FILE):
    plates = set()
    if not os.path.isfile(path):
        return plates
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                value = normalize_plate(line)
                if value:
                    plates.add(value)
    except OSError as exc:
        log("[WARN] Cannot read notebook: %s" % exc)
    return plates


def plate_in_notebook(value, notebook):
    if not value:
        return False
    if value in notebook:
        return True
    if not notebook:
        return False
    best_score = 0.0
    for known in notebook:
        score = fuzz.ratio(value, known)
        if score > best_score:
            best_score = score
    return best_score >= 88.0


def canonical_notebook_value(value, notebook):
    if value in notebook:
        return value, True
    if not notebook:
        return value, False
    best_value, best_score = "", 0.0
    for known in notebook:
        score = fuzz.ratio(value, known)
        if score > best_score:
            best_value, best_score = known, score
    if best_score >= 88.0:
        return best_value, True
    return value, False


def canonical_track_value(value, track):
    """Merge a dirty-frame OCR variant with an existing one-char-near vote."""
    if len(value) != 8:
        return value

    closest = None
    closest_votes = -1

    for previous, votes in track.votes.items():
        if len(previous) != 8:
            continue

        differences = sum(
            current != earlier
            for current, earlier in zip(value, previous)
        )

        if differences <= 1 and votes > closest_votes:
            closest = previous
            closest_votes = votes

    return closest if closest is not None else value


class VehicleTrack:
    def __init__(self, box, class_id, frame_number):
        self.box = box
        self.class_id = class_id
        self.last_seen = frame_number
        self.last_ocr = -10 ** 9
        self.votes = Counter()
        self.best_crop = {}
        self.best_sharpness = {}

    def can_ocr(self, frame_number, interval):
        return frame_number - self.last_ocr >= interval


def get_track(tracks, box, class_id, frame_number):
    best_track, best_iou = None, 0.0
    for track in tracks:
        if track.class_id != class_id:
            continue
        overlap = box_iou(track.box, box)
        if overlap > best_iou:
            best_track, best_iou = track, overlap
    if best_track is None or best_iou < 0.30:
        best_track = VehicleTrack(box, class_id, frame_number)
        tracks.append(best_track)
    best_track.box = box
    best_track.last_seen = frame_number
    return best_track


def save_crop(crop, plate, frame_number, copy_desktop):
    os.makedirs(CROPS_DIR, exist_ok=True)
    name = "%s_frame_%d_%s.jpg" % (datetime.now().strftime("%Y%m%d_%H%M%S"), frame_number, plate)
    path = os.path.join(CROPS_DIR, name)
    if not cv2.imwrite(path, crop):
        log("[WARN] Could not save %s" % path)
        return None
    if copy_desktop:
        try:
            os.makedirs(DESKTOP_DIR, exist_ok=True)
            shutil.copy2(path, os.path.join(DESKTOP_DIR, name))
        except OSError as exc:
            log("[WARN] Desktop copy failed: %s" % exc)
    return path


def parse_streams(stream_value):
    entries = stream_value or []
    if isinstance(entries, str):
        entries = [entries]
    urls = []
    for raw in entries:
        for item in str(raw).split(","):
            item = item.strip()
            if item:
                urls.append(item)
    return urls


def parse_args():
    parser = argparse.ArgumentParser(description="RDK S100 video license-plate watcher")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument(
        "--stream",
        nargs="*",
        default=[],
        help="One or more RTSP/HTTP URL(s); can also be comma separated, takes priority over --video",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=2.0,
        help="Seconds before reconnecting a stream after a read failure",
    )
    parser.add_argument(
        "--rtsp-tcp",
        action="store_true",
        help="Request TCP transport for RTSP streams when FFmpeg supports it",
    )
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--ocr-interval", type=int, default=18)
    parser.add_argument("--confirmations", type=int, default=2)
    parser.add_argument("--score-thres", type=float, default=MODEL_SCORE)
    parser.add_argument("--nms-thres", type=float, default=MODEL_NMS)
    parser.add_argument("--save-unknown", action="store_true")
    parser.add_argument("--copy-desktop", action="store_true")
    parser.add_argument("--debug-ocr", action="store_true")
    parser.add_argument("--pio-pin", type=int, default=DEFAULT_PIO_PIN)
    parser.add_argument("--pio-duration", type=float, default=DEFAULT_PIO_PULSE)
    return parser.parse_args()


def run_source_worker(source, args, notebook, trigger, shared_state):
    try:
        detector = YoloE11Detector(args.model, args.score_thres, args.nms_thres)
    except Exception as exc:
        log("[ERROR] Worker initialization failed for %s: %s" % (source, exc))
        return
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        log("[WARN] Cannot open source: %s" % source)
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    log("[INFO] Source %s FPS: %.2f" % (source, fps))
    log("[INFO] Stream mode: reconnect delay %.1fs" % args.reconnect_delay)

    tracks = []
    reported = set()
    found = []
    frame_number = processed = detections_count = vehicles_count = 0
    candidate_count = ocr_count = 0
    started = time.perf_counter()
    bpu_total = 0.0

    try:
        while not STOP.is_set():
            ok, frame = capture.read()
            if not ok:
                log("[WARN] Stream read failed on %s; reconnecting in %.1fs" % (source, args.reconnect_delay))
                capture.release()
                time.sleep(args.reconnect_delay)
                capture = cv2.VideoCapture(source)
                if capture.isOpened():
                    log("[INFO] Stream reconnected: %s" % source)
                continue

            frame_number += 1
            if frame_number % args.frame_skip:
                continue

            try:
                detections, bpu_ms = detector.detect(frame)
            except Exception as exc:
                log("[ERROR] BPU detection failed for %s at frame %d: %s" % (
                    source, frame_number, exc
                ))
                break
            processed += 1
            bpu_total += bpu_ms
            detections_count += len(detections)
            vehicles = vehicle_detections(detections)
            vehicles_count += len(vehicles)
            tracks = [track for track in tracks if frame_number - track.last_seen <= max(90, args.ocr_interval * 5)]

            for box, class_id, _score in vehicles:
                track = get_track(tracks, box, class_id, frame_number)
                if not track.can_ocr(frame_number, args.ocr_interval):
                    continue
                track.last_ocr = frame_number

                x1, y1, x2, y2 = box
                vehicle = frame[y1:y2, x1:x2]
                candidate = find_plate_candidate(vehicle)
                if candidate is None:
                    continue

                candidate_count += 1
                ocr_count += 1
                values = run_ocr(candidate, args.debug_ocr)
                if not values:
                    continue

                sharpness = image_sharpness(candidate)
                for raw_value in values:
                    value, known = canonical_notebook_value(raw_value, notebook)
                    value = canonical_track_value(value, track)
                    if value in notebook:
                        known = True
                    track.votes[value] += 1
                    if sharpness > track.best_sharpness.get(value, -1.0):
                        track.best_sharpness[value] = sharpness
                        track.best_crop[value] = candidate.copy()

                    votes = track.votes[value]
                    if votes < args.confirmations or value in reported:
                        continue
                    if notebook and not known and not args.save_unknown:
                        log("[OCR] %s votes=%d not in notebook" % (value, votes))
                        continue

                    if plate_in_notebook(value, notebook):
                        with shared_state["lock"]:
                            if value not in shared_state["triggered"]:
                                shared_state["triggered"].add(value)
                                trigger.activate(duration=max(0.1, args.pio_duration))
                                log("[MATCH] source=%s plate=%s votes=%d trigger=PIO%d" % (
                                    source, value, votes, args.pio_pin
                                ))

                    path = save_crop(track.best_crop[value], value, frame_number, args.copy_desktop)
                    if path is None:
                        continue
                    reported.add(value)
                    found.append(value)
                    log("[FOUND] source=%s plate=%s votes=%d frame=%d known=%s saved=%s" % (
                        source, value, votes, frame_number, known, path
                    ))

            if frame_number % (args.frame_skip * 10) == 0:
                position = "%d/live" % frame_number if source else "%d/%d" % (frame_number, total_frames)
                log("[PROGRESS] source=%s frame=%s BPU=%.1fms vehicles=%d candidates=%d OCR=%d found=%d" % (
                    source, position, bpu_ms, vehicles_count,
                    candidate_count, ocr_count, len(found)
                ))
    finally:
        capture.release()

    elapsed = time.perf_counter() - started
    average_bpu = bpu_total / processed if processed else 0.0
    log("[DONE] source=%s frames=%d processed=%d detections=%d vehicles=%d plate_candidates=%d OCR_calls=%d found=%d avg_BPU=%.2fms elapsed=%.1fs" % (
        source, frame_number, processed, detections_count, vehicles_count, candidate_count,
        ocr_count, len(found), average_bpu, elapsed
    ))
    log("[FOUND LIST] source=%s %s" % (source, ", ".join(found) if found else "none"))


def main():
    args = parse_args()
    args.frame_skip = max(1, args.frame_skip)
    args.ocr_interval = max(1, args.ocr_interval)
    args.confirmations = max(1, args.confirmations)
    args.reconnect_delay = min(30.0, max(0.2, args.reconnect_delay))
    args.pio_duration = max(0.05, float(args.pio_duration))

    stream_sources = parse_streams(args.stream)
    is_stream = bool(stream_sources)
    source = stream_sources[0] if stream_sources else args.video

    if not is_stream and not os.path.isfile(source):
        raise SystemExit("Video not found: %s" % source)
    if not os.path.isfile(args.model):
        raise SystemExit("Model not found: %s" % args.model)

    if args.rtsp_tcp and any(url.lower().startswith("rtsp://") for url in (stream_sources or [source])):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    # Headless / SSH / PuTTY sessions do not require a local X display for RTSP capture.
    os.environ.setdefault("DISPLAY", "")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    notebook_path = ensure_notebook_file(NOTEBOOK_FILE)
    notebook = load_notebook(notebook_path)
    log("[INFO] Starting %s: %s" % (
        "stream" if is_stream else "video",
        source if is_stream else source,
    ))
    log("[INFO] Notebook plates: %d" % len(notebook))
    log("[INFO] Tesseract: %s" % pytesseract.get_tesseract_version())
    log("[INFO] frame-skip=%d OCR interval=%d confirmations=%d" % (
        args.frame_skip, args.ocr_interval, args.confirmations
    ))
    log("[INFO] PIO trigger pin=%d pulse=%.2fs" % (args.pio_pin, args.pio_duration))

    try:
        trigger = PioTrigger(pin=args.pio_pin)
    except (OSError, IOError) as exc:
        log("[WARN] PIO pin %d unavailable: %s" % (args.pio_pin, exc))
        trigger = NoopTrigger()
    shared_state = {"lock": threading.Lock(), "triggered": set()}

    if is_stream and len(stream_sources) > 1:
        threads = []
        for stream_url in stream_sources:
            log("[INFO] Starting concurrent stream thread: %s" % stream_url)
            thread = threading.Thread(target=run_source_worker, args=(stream_url, args, notebook, trigger, shared_state), daemon=True)
            thread.start()
            threads.append(thread)
        while not STOP.is_set():
            time.sleep(0.5)
        for thread in threads:
            thread.join(timeout=1.0)
        return

    run_source_worker(source, args, notebook, trigger, shared_state)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)
    main()