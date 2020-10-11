import numpy as np

from seldon_core.user_model import SeldonComponent
from tflite_runtime import interpreter as tflite
from typing import List, Dict, Iterable

INPUT_TENSOR_NAME = "data_1"
OUTPUT_TENSOR_NAME = "cls_branch_concat_1/concat"


class TFLiteServer(SeldonComponent):
    def __init__(self, model_uri="./models/face_mask_detection.tflite"):
        self._model_uri = model_uri

    def load(self):
        self._interpreter = tflite.Interpreter(model_path=self._model_uri)
        self._interpreter.allocate_tensors()

        # Obtain input tensor index
        input_tensors = self._interpreter.get_input_details()
        self._input_tensor_index = self._get_tensor_index(
            input_tensors, INPUT_TENSOR_NAME
        )

        # Obtain output tensor index
        output_tensors = self._interpreter.get_output_details()
        self._output_tensor_index = self._get_tensor_index(
            output_tensors, OUTPUT_TENSOR_NAME
        )

    def _get_tensor_index(self, tensors: List[Dict], tensor_name: str) -> int:
        for tensor in tensors:
            if tensor["name"] == tensor_name:
                return tensor["index"]

        raise RuntimeError(f"Tensor name not found: {tensor_name}")

    def predict(self, X: np.ndarray, names: Iterable[str], meta: Dict = None):
        # Force input to be np.float32
        img = np.float32(X)

        # NOTE: This is not thread-safe!
        self._interpreter.set_tensor(self._input_tensor_index, img)
        self._interpreter.invoke()

        output = self._interpreter.get_tensor(self._output_tensor_index)

        return output
