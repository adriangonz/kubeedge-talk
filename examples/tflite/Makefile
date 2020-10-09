VERSION=0.1.0
IMAGE=adriangonz/mlserver-tflite:${VERSION}

.PHONY: build push 

models/face_mask_detection.tflite: 
	mkdir -p models
	wget \
		-P models \
		https://github.com/AIZOOTech/FaceMaskDetection/raw/master/models/face_mask_detection.tflite

build: models/face_mask_detection.tflite
	DOCKER_CLI_EXPERIMENTAL=enabled \
		docker buildx build \
			. \
			--load \
			--platform linux/arm/v7 \
			-t ${IMAGE}

push:
	docker push ${IMAGE}
