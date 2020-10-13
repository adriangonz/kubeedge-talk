import logging
import requests
import time
import os
import numpy as np

from picamera import PiCamera
from picamera.array import PiRGBArray
from typing import Tuple


DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MODEL_SERVER = os.getenv("MODEL_SERVER", default="http://model:9000")
PIXEL_FORMAT = "RGB"
CAMERA_RESOLUTION = (260, 260)
CAMERA_WARMUP_SECONDS = 2
CONFIDENCE_THRESHOLD = 0.5


def _setup_logger():
    log_level = logging.INFO
    if DEBUG:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)


def _get_camera() -> PiCamera:
    logging.info(f"Accessing camera with {CAMERA_RESOLUTION} resolution")
    camera = PiCamera()
    camera.resolution = CAMERA_RESOLUTION
    # Start a preview and let the camera warm up for 2 seconds
    logging.info("Waiting for camera to warm up...")
    camera.start_preview()
    time.sleep(CAMERA_WARMUP_SECONDS)

    logging.info("Obtained camera handle!")
    return camera


def _save_frame(frame: np.ndarray):
    pass


def _run_inference(frame: np.ndarray) -> np.ndarray:
    logging.debug(f"Running inference in frame with shape {frame.shape}...")

    # Normalise pixels to [0-1] range
    batch = np.expand_dims(frame, axis=0) / 255.0
    payload = {"data": {"ndarray": batch.tolist()}}
    endpoint = f"{MODEL_SERVER}/api/v1.0/predictions"

    logging.debug(f"Sending request to inference endpoint {endpoint}...")
    response = requests.post(endpoint, json=payload)
    if not response.ok:
        raise RuntimeError("Invalid frame")

    json_response = response.json()
    confidences = np.array(json_response["data"]["array"])
    logging.debug(f"Obtained prediction with shape {confidences.shape}")

    # Filter out low-confidence predictions
    max_confidences = np.max(confidences, axis=2)
    classes = np.argmax(confidences, axis=2)
    high_confidence = np.where(max_confidences > CONFIDENCE_THRESHOLD)

    return classes[high_confidence]


def _update_leds(y_pred: np.ndarray):
    logging.debug("Updating LEDs...")

    without_mask = np.count_nonzero(y_pred)
    with_mask = len(y_pred) - without_mask
    logging.debug(f"Detected {without_mask} persons without mask")
    logging.debug(f"Detected {with_mask} persons with mask")

    # TODO: Update LEDs


def main():
    _setup_logger()
    camera = _get_camera()
    frame = PiRGBArray(camera)

    logging.info("Starting capture loop... Smile!")
    for _ in camera.capture_continuous(frame, PIXEL_FORMAT.lower()):
        if DEBUG:
            _save_frame(frame.array)

        y_pred = _run_inference(frame.array)
        _update_leds(y_pred)
        # Truncate to re-use
        # https://picamera.readthedocs.io/en/release-1.13/api_array.html#pirgbarray
        frame.truncate(0)


if __name__ == "__main__":
    main()
