from seldon_core.user_model import SeldonComponent
from tflite_runtime import interpreter as tflite


class TFLiteServer(SeldonComponent):
    def __init__(self, model_uri="./models/face_mask_detection.tflite"):
        self._model_uri = model_uri

    def load(self):
        self._interpreter = tflite.Interpreter(model_path=self._model_uri)
        self._interpreter.allocate_tensors()

    def predict(self, X):
        # NOTE: This is not thread-safe!
        self._interpreter.set_tensor("input0", X)
        self._interpreter.invoke()

        output_name = "output0"
        output = self._interpreter.get_tensor(output_name)

        return output
