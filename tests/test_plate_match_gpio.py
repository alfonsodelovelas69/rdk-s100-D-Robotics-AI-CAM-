import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def load_main_module():
    stub_module("hbm_runtime", HB_HBMRuntime=object)
    stub_module("cv2",
        COLOR_BGR2GRAY=0,
        CV_64F=0,
        MORPH_BLACKHAT=0,
        MORPH_RECT=0,
        THRESH_BINARY=0,
        THRESH_OTSU=0,
        ADAPTIVE_THRESH_GAUSSIAN_C=0,
        BORDER_CONSTANT=0,
        INTER_CUBIC=0,
        createCLAHE=lambda *args, **kwargs: types.SimpleNamespace(apply=lambda image: image),
        cvtColor=lambda *args, **kwargs: None,
        Laplacian=lambda *args, **kwargs: types.SimpleNamespace(var=lambda: 0.0),
        morphologyEx=lambda *args, **kwargs: None,
        threshold=lambda *args, **kwargs: (0, None),
        adaptiveThreshold=lambda *args, **kwargs: None,
        bitwise_not=lambda *args, **kwargs: None,
        bilateralFilter=lambda *args, **kwargs: None,
        getStructuringElement=lambda *args, **kwargs: None,
        findContours=lambda *args, **kwargs: ([], None),
        boundingRect=lambda *args, **kwargs: (0, 0, 0, 0),
        resize=lambda *args, **kwargs: None,
        imwrite=lambda *args, **kwargs: True,
        VideoCapture=lambda *args, **kwargs: None,
        CAP_PROP_FPS=0,
        CAP_PROP_FRAME_COUNT=0,
    )
    stub_module("numpy",
        arange=lambda *args, **kwargs: [],
        concatenate=lambda *args, **kwargs: [],
        log=lambda *args, **kwargs: 0,
        float32=float,
    )
    stub_module("pytesseract", image_to_string=lambda *args, **kwargs: "AA1234BB")
    stub_module("rapidfuzz", fuzz=types.SimpleNamespace(ratio=lambda a, b: 100.0 if a == b else 70.0))

    util_pkg = types.ModuleType("utils")
    util_pkg.__path__ = []
    sys.modules.setdefault("utils", util_pkg)

    preprocess = types.ModuleType("utils.preprocess_utils")
    preprocess.resized_image = lambda *args, **kwargs: None
    preprocess.bgr_to_nv12_planes = lambda *args, **kwargs: (None, None)
    sys.modules["utils.preprocess_utils"] = preprocess

    postprocess = types.ModuleType("utils.postprocess_utils")
    postprocess.dequantize_outputs = lambda *args, **kwargs: {}
    postprocess.filter_classification = lambda *args, **kwargs: ([], [], [])
    postprocess.decode_boxes = lambda *args, **kwargs: []
    postprocess.NMS = lambda *args, **kwargs: []
    postprocess.scale_coords_back = lambda *args, **kwargs: []
    sys.modules["utils.postprocess_utils"] = postprocess

    spec = importlib.util.spec_from_file_location("main_mod", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["main_mod"] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_plate_matches_notebook_values(tmp_path):
    module = load_main_module()
    notebook = {"AA1234BB"}
    assert module.normalize_plate("AA1234BB") == "AA1234BB"
    assert module.plate_in_notebook("AA1234BB", notebook) is True
    assert module.plate_in_notebook("AA1234BC", notebook) is False


def test_gpio_trigger_exports_and_pulses(tmp_path):
    module = load_main_module()
    gpio_dir = tmp_path / "gpio"
    gpio_dir.mkdir()

    class FakeGPIO:
        def __init__(self, pin, active_high=True):
            self.pin = pin
            self.active_high = active_high
            self.value_path = gpio_dir / f"gpio{pin}" / "value"
            self.value_path.parent.mkdir(parents=True, exist_ok=True)
            self.value_path.write_text("0")

        def set_value(self, value):
            self.value_path.write_text(str(value))

    trigger = module.PioTrigger(pin=17, active_high=True, sysfs_root=gpio_dir, gpio_factory=FakeGPIO)
    trigger.activate(duration=0.01)
    assert trigger.last_state == 1


def test_video_watch_supports_multiple_rtsp_sources():
    spec = importlib.util.spec_from_file_location("video_watch_mod", ROOT / "video_plate_watch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    streams = module.parse_streams(['rtsp://cam1', 'rtsp://cam2'])
    assert streams == ['rtsp://cam1', 'rtsp://cam2']

    streams = module.parse_streams('rtsp://cam1,rtsp://cam2')
    assert streams == ['rtsp://cam1', 'rtsp://cam2']


def test_video_watch_alternates_streams_by_processed_frame_count():
    spec = importlib.util.spec_from_file_location("video_watch_scheduler_mod", ROOT / "video_plate_watch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scheduler = module.FrameTurnScheduler(stream_count=2, frames_per_turn=3)
    assert scheduler.wait_turn(0) is True
    for _ in range(3):
        scheduler.complete_frame(0)
    assert scheduler.wait_turn(1) is True

    scheduler.deactivate(1)
    assert scheduler.wait_turn(0) is True
