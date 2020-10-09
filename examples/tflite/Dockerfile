FROM arm32v7/python:3.7.8-slim

WORKDIR /usr/src/app

ENV PATH="/root/.cargo/bin:${PATH}"

# Install Git.
# Necessary to install the `mlserver` dependency from the repo, (since it's not
# available in PyPi yet).
RUN apt-get update && \
    apt-get install -y git curl

RUN pip install --upgrade wheel setuptools pip

# Install Rust.
# Necessary to compile `orjson`, which lacks a wheel for ARM.
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

COPY requirements.txt ./
RUN pip install \
    -r requirements.txt \
    --extra-index-url=https://piwheels.org/simple

COPY . .

CMD mlserver start .
