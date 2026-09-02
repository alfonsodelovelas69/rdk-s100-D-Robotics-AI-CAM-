#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RDK S100: YOLOE11 HBM vehicle detection + strict Ukrainian plate OCR.
   Паралельна обробка кількох потоків/файлів, walking-camera режим та PIO тригер.
   Зміна: модель ініціалізується централізовано в main() і передається в worker-threads
         (з блокуванням навколо model.run() для безпечного шерингу BPU).
"""

import argparse
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
DEFAULT_PIO_PIN = 40
DEFAULT_PIO_PULSE = 0.35

# Confirmed IDs from the prompt-free YOLOE11 model on this S100.
VEHICLE_IDS = {726, 4329}  # car, van

MODEL_SCORE = 0.40
MODEL_NMS = 0.70
MIN_VEHICLE_SCORE = 0.45
MIN_VEHICLE_WIDTH = 100
MIN_VEHICLE_HEIGHT = 60

MIN_PLATE_WIDTH = 55
MIN_PLATE_HEIGHT = 12
MIN_PLATE_RATIO = 2.2
MAX_PLATE_RATIO = 7.0
MIN_PLATE_SHARPNESS = 28.0

STOP = threading.Event()


def on_stop(_signum, _frame):
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


# --- GPIO / PIO trigger (sysfs) ---
class PioTrigger:
    """Simple sysfs GPIO pulse for one PIO output."""

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
            try:
                self.gpio.set_value(active_value)
                if duration > 0:
                    time.sleep(duration)
                self.gpio.set_value(idle_value)
                self.last_state = idle_value
            except Exception as exc:
                log("[WARN] PIO activation failed: %s" % exc)


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


# --- utility functions (boxes, crop validity, iou, ocr prep) ---
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


# --- Detector wrapper (same as sample) ---
class YoloE11Detector:
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
        # Lock to serialize model.run() calls when sharing the detector between threads
        self.lock = threading.Lock()

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

        # Serialize access to model.run() to avoid concurrent BPU calls on the same handle
        with self.lock:
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


# --- plate finding for walking/static camera (uses rectify for walking) ---
def order_quad(points):
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def rectify_candidate(roi, contour):
    perimeter = cv2.arcLength(contour, True)
    approximate = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    if len(approximate) == 4:
        source = approximate.reshape(4, 2)
    else:
        source = cv2.boxPoints(cv2.minAreaRect(contour))
    source = order_quad(source)
    top = np.linalg.norm(source[1] - source[0])
    bottom = np.linalg.norm(source[2] - source[3])
    right = np.linalg.norm(source[2] - source[1])
    left = np.linalg.norm(source[3] - source[0])
    target_width = int(round(max(top, bottom)))
    target_height = int(round(max(right, left)))
    if target_width < 20 or target_height < 8:
        return None
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(roi, transform, (target_width, target_height))


def find_plate_candidate(vehicle, walking_camera=False):
    if not crop_valid(vehicle):
        return None
    height, width = vehicle.shape[:2]
    if width < MIN_VEHICLE_WIDTH or height < MIN_VEHICLE_HEIGHT:
        return None
    roi_y = int(height * (0.32 if walking_camera else 0.42))
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
        min_ratio = 1.7 if walking_camera else MIN_PLATE_RATIO
        max_ratio = 9.0 if walking_camera else MAX_PLATE_RATIO
        min_width = 35 if walking_camera else MIN_PLATE_WIDTH
        min_height = 8 if walking_camera else MIN_PLATE_HEIGHT
        min_area = 350 if walking_camera else 700
        max_area_fraction = 0.13 if walking_camera else 0.10

        if not (min_ratio <= ratio <= max_ratio):
            continue
... (file continues)