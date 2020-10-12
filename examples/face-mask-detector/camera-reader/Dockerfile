FROM arm32v7/python:3.7.8-slim

WORKDIR /usr/src/app

# Install native subdeps of Python deps.
# - `libatlas`: necessary for `numpy`
RUN apt-get update \
    && apt-get install -y libatlas-base-dev

RUN pip install --upgrade wheel setuptools pip

COPY requirements.txt ./
RUN pip install \
    -r requirements.txt \
    --extra-index-url=https://piwheels.org/simple

COPY . .

CMD [ "python", "./camera.py" ]
