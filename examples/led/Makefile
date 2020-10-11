VERSION := 0.1.0
IMAGE := adriangonz/kubeedge-led-example:${VERSION}
PLATFORM := linux/arm/v7

.PHONY: build push

build:
	docker build \
		. \
		--platform ${PLATFORM} \
		-t ${IMAGE}

push:
	docker push ${IMAGE}
