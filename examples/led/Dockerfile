FROM arm32v7/python:3.7.8-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "./led.py" ]
