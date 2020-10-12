import numpy as np

from seldon_core.user_model import SeldonComponent
from tflite_runtime import interpreter as tflite
from typing import List, Dict, Iterable


class TFLiteServer(SeldonComponent):
    def __init__(
        self,
        model_uri="./models/face_mask_detection.tflite",
        input_tensor_name="data_1",
        output_tensor_name="cls_branch_concat_1/concat",
    ):
        self._model_uri = model_uri
        self._input_tensor_name = input_tensor_name
        self._output_tensor_name = output_tensor_name

    def load(self):
        self._interpreter = tflite.Interpreter(model_path=self._model_uri)
        self._interpreter.allocate_tensors()

        # Obtain input tensor index
        input_tensors = self._interpreter.get_input_details()
        self._input_tensor_index = self._get_tensor_index(
            input_tensors, self._input_tensor_name
        )

        # Obtain output tensor index
        output_tensors = self._interpreter.get_output_details()
        self._output_tensor_index = self._get_tensor_index(
            output_tensors, self._output_tensor_name
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
