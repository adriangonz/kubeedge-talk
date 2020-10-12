VERSION := 0.1.0
ARCH := arm
IMAGE := adriangonz/camera-reader:${VERSION}-${ARCH}
PLATFORM := linux/arm/v7

.PHONY: build push 

build: 
	docker build \
		. \
		--platform ${PLATFORM} \
		-t ${IMAGE}

push:
	docker push ${IMAGE}
