import logging
import requests
import time
import numpy as np

from picamera import PiCamera
from picamera.array import PiRGBArray
from typing import Tuple


DEBUG = False
PIXEL_FORMAT = "RGB"
FACEMASK_DETECTOR_SERVER = "http://model:9000"


def _get_camera(resolution: Tuple[int, int] = (640, 480)) -> PiCamera:
    logging.info(f"Accessing camera with {resolution} resolution")
    camera = PiCamera()
    camera.resolution = resolution
    # Start a preview and let the camera warm up for 2 seconds
    logging.info("Waiting for camera to warm up...")
    camera.start_preview()
    time.sleep(2)

    logging.info("Obtained camera handle!")
    return camera


def _save_frame(frame: np.ndarray):
    pass


def _run_inference(frame: np.ndarray) -> np.ndarray:
    logging.debug(f"Running inference in frame with shape {frame.shape}...")
    batch = np.expand_dims(frame, axis=0)
    payload = {"data": {"ndarray": batch.tolist()}}
    endpoint = f"{FACEMASK_DETECTOR_SERVER}/api/v1.0/predictions"

    logging.debug(f"Sending request to inference endpoint {endpoint}...")
    response = requests.post(endpoint, json=payload)
    if not response.ok:
        raise RuntimeError("Invalid frame")

    y_pred = response.json()
    logging.debug(f"Obtained prediction with shape {y_pred.shape}")

    # TODO: Filter out lower than threshold and keep classes

    return y_pred


def _update_leds():
    logging.debug("Updating LEDs...")
    pass


def main():
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
