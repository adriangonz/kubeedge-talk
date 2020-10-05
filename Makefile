KUBEEDGE_VERSION := v1.4.0
ARCH := amd64
KUBEEDGE_FNAME := kubeedge-${KUBEEDGE_VERSION}-linux-${ARCH}

.PHONY: install-dev local-cluster

download-kubeedge:
	mkdir -p tempdir
	wget \
		--directory-prefix ./tempdir \
		https://github.com/kubeedge/kubeedge/releases/download/${KUBEEDGE_VERSION}/${KUBEEDGE_FNAME}.tar.gz
	tar \
		--directory ./tempdir \
		-xvzf tempdir/${KUBEEDGE_FNAME}.tar.gz
	cp -r ./tempdir/${KUBEEDGE_FNAME}/* ./kubeedge
	rm -rf tempdir

local-cluster:
	kind create cluster \
		--config=config/kind.yaml \
		--name kubeedge

kubeedge-cloud:
	kubectl apply -f ./kubeedge/crds/*

