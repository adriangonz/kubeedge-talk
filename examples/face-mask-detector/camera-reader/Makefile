VERSION := 0.1.0
ARCH := arm
IMAGE := adriangonz/camera-reader:${VERSION}-${ARCH}
PLATFORM := linux/arm/v7

.PHONY: build push 

build: 
	# Can't build image in x86_64 hosts (even with qemu) due to:
	# https://github.com/docker/buildx/issues/395
	docker build \
		. \
		--platform ${PLATFORM} \
		-t ${IMAGE}

push:
	docker push ${IMAGE}
