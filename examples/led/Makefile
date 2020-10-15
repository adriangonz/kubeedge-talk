VERSION := 0.1.0-arm
IMAGE := adriangonz/kubeedge-led-example:${VERSION}
PLATFORM := linux/arm/v7

.PHONY: build push

README.md: README.ipynb
	jupyter nbconvert \
		README.ipynb \
		--ClearOutputPreprocessor.enabled=True \
		--to markdown \
		--output README.md


build:
	docker build \
		. \
		--platform ${PLATFORM} \
		-t ${IMAGE}

push:
	docker push ${IMAGE}
