import requests
import time
import numpy as np

from PIL import Image
from picamera import PiCamera
from picamera.array import PiRGBArray
from typing import Tuple


DEBUG = True
PIXEL_FORMAT = "RGB"
FACEMASK_DETECTOR_SERVER = "http://model:9000"


def _get_camera(resolution: Tuple[int, int] = (640, 480)) -> PiCamera:
    camera = PiCamera()
    camera.resolution = (640, 480)
    # Start a preview and let the camera warm up for 2 seconds
    camera.start_preview()
    time.sleep(2)

    return camera


def _save_frame(frame: np.ndarray):
    img = Image.fromarray(frame, PIXEL_FORMAT)
    img.save("last-image.png")


def _run_inference(frame: np.ndarray) -> np.ndarray:
    batch = np.expand_dims(frame, axis=0)
    payload = {"data": {"ndarray": batch.tolist()}}
    endpoint = f"{FACEMASK_DETECTOR_SERVER}/api/v1.0/predictions"

    response = requests.post(endpoint, json=payload)
    if not response.ok:
        raise RuntimeError("Invalid frame")

    y_pred = response.json()

    # TODO: Filter out lower than threshold and keep classes

    return y_pred


def _update_leds():
    pass


def main():
    camera = _get_camera()
    frame = PiRGBArray(camera)

    for _ in camera.capture_continuous(frame, PIXEL_FORMAT.lower()):
        if DEBUG:
            _save_frame(frame)

        y_pred = _run_inference(frame)
        _update_leds(y_pred)
        # Truncate to re-use
        # https://picamera.readthedocs.io/en/release-1.13/api_array.html#pirgbarray
        frame.truncate(0)


if __name__ == "__main__":
    main()
