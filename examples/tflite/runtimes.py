from mlserver import MLModel, types
from tflite_runtime import interpreter as tflite


class TFLiteRuntime(MLModel):
    async def load(self) -> bool:
        model_path = self._settings.parameters.uri
        self._interpreter = tflite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        return True

    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        input_tensor = payload.inputs[0]

        # NOTE: This is not thread-safe!
        self._interpreter.set_tensor(input_tensor.name, input_tensor.data)
        self._interpreter.invoke()

        output_name = payload.outputs[0].name
        output = self._interpreter.get_tensor(output_name)

        return types.InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                types.ResponseOutput(
                    name=output_name,
                    shape=output.shape,
                    datatype="FP32",
                    data=output.tolist(),
                )
            ],
        )
