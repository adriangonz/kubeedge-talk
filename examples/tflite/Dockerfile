FROM arm32v7/python:3.7.8-slim

WORKDIR /usr/src/app

ARG API_TYPE="REST"
ARG SERVICE_TYPE="MODEL"

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
