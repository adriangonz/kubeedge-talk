VERSION=0.1.0
IMAGE=adriangonz/kubeedge-led-example:${VERSION}

build:
	DOCKER_CLI_EXPERIMENTAL=enabled \
		docker buildx build \
			. \
			--load \
			--platform linux/arm/v7 \
			-t ${IMAGE}

push: build
	docker push ${IMAGE}
