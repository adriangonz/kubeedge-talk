# Inference with TFLite on Edge devices

This example walks through how to deploy a [TFLite model](https://www.tensorflow.org/lite/guide) in an edge device using Seldon Core.
In particular, we will deploy a face mask detector in a Raspberry Pi, which will run inference locally from a camera feed.

The face mask detector has already been pre-trained by the team at [Aizoo Tech team](https://aizoo.com) who has kindly put it available on the [AIZOOTech/FaceMaskDetection](https://github.com/AIZOOTech/FaceMaskDetection) repository. **Massive thanks to them!**


## Pre-requisites

### Seldon Core

The example assumes that Seldon Core has already been installed in the cluster.
If it hasn't, you can follow the [setup instructions in the Seldon Core documentation](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/install.html).

It's also worth noting that some steps below assume that Seldon Core has been installed using Helm with name `seldon-core` in the `seldon-system` namespace.
This is not a hard requirement though, and it should be simple to adapt those steps in case it has been installed in a different location.

### KubeEdge

The example assumes that KubeEdge has already been installed in the cluster, exposing the `cloudcore` component so that it's accessible from an edge device.

### Edge Device

The example assumes that the edge device is a Raspberry Pi with an ARMv7 architecture of 32 bits.
In particular, the example has been tested with a Raspberry Pi 3 Model B V1.2.

We will assume that the device has been pre-synced through KubeEdge and that it's already visible as a node in the cluster with name `raspberry`. 

## TFLite Inference Server

Out of the box, Seldon Core doesn't offer support for TFLite models (particularly under an ARM architecture).
However, as we shall see, it's fairly simple to leverage their support for [custom inference servers](https://docs.seldon.io/projects/seldon-core/en/latest/servers/custom.html).

### Implementing Model Runtime

Firstly, we'll extend the `SeldonComponent` interface to create a new model runtime.
The methods that we'll want to extend are:

- `load()`: responsible for loading our TFLite model
- `predict()`: responsible for running inference against our model

Note that we also want to parametrise the following model-specific details:

- Where are model weights loaded from (i.e. `model_uri`).
- On which tensor should we load the input data (i.e. `input_tensor_name`).
- From which tensor should we read the model's output (i.e. `output_tensor_name`).


```python
%%writefile ./tfliteserver/TFLiteServer.py
import os
import glob
import numpy as np

from seldon_core.user_model import SeldonComponent
from tflite_runtime import interpreter as tflite
from typing import List, Dict, Iterable

TFLITE_EXT = "*.tflite"


class TFLiteServer(SeldonComponent):
    def __init__(
        self,
        model_uri: str,
        input_tensor_name: str,
        output_tensor_name: str,
    ):
        self._model_uri = model_uri
        self._input_tensor_name = input_tensor_name
        self._output_tensor_name = output_tensor_name

    def load(self):
        model_path = self._get_model_path()
        self._interpreter = tflite.Interpreter(model_path=model_path)
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

    def _get_model_path(self) -> str:
        # Search for *.tflite files to load
        pattern = os.path.join(self._model_uri, TFLITE_EXT)
        model_paths = glob.glob(pattern)
        if not model_paths:
            raise RuntimeError(f"No models found at {self._model_uri}")

        return model_paths[0]

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

```

### Building runtime image

The next step will be building our runtime into a Docker image.
It's worth mentioning that the new image has to be compatible with an ARM architecture, therefore we won't be able to use the [`s2i` facilities provided by Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/python/python_wrapping_s2i.html) out of the box.
However, we can put up a together a `Dockerfile` based on [Seldon's documentation to wrap models using Docker directly](https://docs.seldon.io/projects/seldon-core/en/latest/python/python_wrapping_docker.html).


```python
%%writefile ./tfliteserver/Dockerfile
FROM arm32v7/python:3.7.8-slim

WORKDIR /usr/src/app

ENV API_TYPE="REST"
ENV SERVICE_TYPE="MODEL"

# Install native subdeps of Python deps.
# - `libatlas`: necessary for `numpy`
# - `libatomic1`: necessary for `grpcio`
RUN apt-get update \
    && apt-get install -y libatlas-base-dev libatomic1

RUN pip install --upgrade wheel setuptools pip

COPY requirements.txt ./
RUN pip install \
    -r requirements.txt \
    --extra-index-url=https://piwheels.org/simple

COPY . .

CMD seldon-core-microservice TFLiteServer $API_TYPE --service-type $SERVICE_TYPE 

```

Note that to build this image, we'll need a host able to build `arm` images.
In Linux, we can achieve this using QEMU.


```python
!docker build \
    ./tfliteserver \
    --platform linux/arm/v7 \
    -t adriangonz/tfliteserver:0.1.0-arm
!docker push adriangonz/tfliteserver:0.1.0-arm
```

### Adding new inference server to Seldon Core

Lastly, we'll need to configure the new runtime as a [custom inference server within Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/servers/custom.html).
Note that Seldon Core allows you to override the image used for your model at the `SeldonDeployment` level, thus **this step is not needed**.
However, it leads to higher reusability.

For our example, we will configure our new runtime under the `TFLITE_SERVER` key, by adding the following to the `predictor_servers` key in the `seldon-config` configmap:

```json
{
     "TFLITE_SERVER": {
         "rest": {
             "defaultImageVersion": "0.1.0-arm",
             "image": "adriangonz/tfliteserver"
         }
     }
}
```

Note that this can be added as part of the values of the Helm chart when installing Seldon Core.
If we assume that the chart has been installed under the `seldon-core` name in the `seldon-system` namespace, we can do:


```python
!helm upgrade seldon-core seldonio/seldon-core-operator \
    -n seldon-system \
    --reuse-values \
    --set 'predictor_servers.TFLITE_SERVER.rest.defaultImageVersion=0.1.0-arm' \
    --set 'predictor_servers.TFLITE_SERVER.rest.image=adriangonz/tfliteserver'
```

## Model Initialiser Image

The model initialiser is part of the Seldon Core architecture.
It's responsible for downloading any model artifacts at init time, which will then be exposed to the actual model deployment.
This functionality is wrapped as a Docker image, which currently only supports the `x86` architecture.

Therefore, to leverage it on our edge deployment, we will need to re-build it using an ARM-compatible base.
You can find a pre-built compatible image under tag `adriangonz/storage-initializer:v0.4.0-arm`.
The only extra steep necessary will be to configure Seldon Core to use this image instead of the default one.

As before, we can do this by changing the `seldon-config` namespace.
We can do this by modifying the values of the Seldon Core Helm chart.
If we assume that the chart has been installed under the `seldon-core` name in the `seldon-system` namespace, we could run:


```python
!helm upgrade seldon-core seldonio/seldon-core-operator \
    -n seldon-system \
    --reuse-values \
    --set 'storageInitializer.image=adriangonz/storage-initializer:v0.4.0-arm'
```

## Deploy model

Now that we've got the different components set up, it should be possible to deploy our model instantiating a `SeldonDeployment` resource.
This resource needs to specify that our machine learning deployment will:

- Leverage the `TFLITE_SERVER` inference server.
- Load the model weights from the [AIZOOTech/FaceMaskDetection repo](https://github.com/AIZOOTech/FaceMaskDetection/raw/master/models/face_mask_detection.tflite).
- Be deployed to the node named `raspberry`, which is an edge device with an ARM architecture.
- Specify that the model input goes into the tensor named `data_1` and the model output comes from the tensor named `cls_branch_concat_1/concat` (these are model-specific details which we've parametrised in out TFLite runtime).

This can be done easily with Seldon Core as:


```python
%%writefile ./charts/seldondeployment-face-mask-detector.yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: face-mask-detector
  namespace: examples
spec:
  predictors:
    - annotations:
        seldon.io/no-engine: "true"
      graph:
        name: model
        implementation: TFLITE_SERVER
        modelUri: 'https://github.com/AIZOOTech/FaceMaskDetection/raw/master/models/face_mask_detection.tflite'
        parameters:
            - name: input_tensor_name
              value: data_1
              type: STRING
            - name: output_tensor_name
              value: cls_branch_concat_1/concat
              type: STRING
        children: []
      componentSpecs:
        - spec:
            nodeName: raspberry
      name: default

```


```python
!kubectl apply -f ./charts/seldondeployment-face-mask-detector.yaml
```


```python

```
