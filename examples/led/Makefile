VERSION=0.1.0
IMAGE=adriangonz/kubeedge-led-example:${VERSION}

build:
	docker build \
		. \
		--platform linux/arm/v7 \
		-t ${IMAGE}

push:
	docker push ${IMAGE}
