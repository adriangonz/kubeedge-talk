VERSION=0.1.0
IMAGE=adriangonz/mlserver-tflite:${VERSION}

.PHONY: build push 

models/face_mask_detection.tflite: 
	mkdir -p models
	wget \
		-P models \
		https://github.com/AIZOOTech/FaceMaskDetection/raw/master/models/face_mask_detection.tflite

build: models/face_mask_detection.tflite
	# Can't build image in x86_64 hosts (even with qemu) due to:
	# https://github.com/docker/buildx/issues/395
	docker build \
		. \
		--platform linux/arm/v7 \
		-t ${IMAGE}

push:
	docker push ${IMAGE}
