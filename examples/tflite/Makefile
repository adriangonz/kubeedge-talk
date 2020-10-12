VERSION := 0.1.0
ARCH := arm
IMAGE := adriangonz/tfliteserver:${VERSION}-${ARCH}
PLATFORM := linux/arm/v7

.PHONY: build push 

models/face_mask_detection.tflite: 
	mkdir -p models
	wget \
		-P models \
		https://github.com/AIZOOTech/FaceMaskDetection/raw/master/models/face_mask_detection.tflite

README.md: README.ipynb
	jupyter nbconvert \
		README.ipynb \
		--ClearOutputPreprocessor.enabled=True \
		--to markdown \
		--output README.md

build: models/face_mask_detection.tflite
	# Can't build image in x86_64 hosts (even with qemu) due to:
	# https://github.com/docker/buildx/issues/395
	docker build \
		. \
		--platform ${PLATFORM} \
		-t ${IMAGE}

push:
	docker push ${IMAGE}

install-dev:
	pip install -r requirements-dev.txt
