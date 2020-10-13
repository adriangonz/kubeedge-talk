VERSION := 0.1.0
ARCH := arm
IMAGE := adriangonz/tfliteserver:${VERSION}-${ARCH}
PLATFORM := linux/arm/v7

.PHONY: build push install-dev

models/face_mask_detection.tflite: 
	mkdir -p models
	wget \
		-P models \
		https://github.com/AIZOOTech/FaceMaskDetection/raw/master/models/face_mask_detection.tflite

build: 
	docker build \
		. \
		--platform ${PLATFORM} \
		-t ${IMAGE}

push:
	docker push ${IMAGE}

install-dev:
	pip install -r requirements-dev.txt
