KUBEEDGE_VERSION := v1.4.0
ARCH := arm
KEADM_FNAME := keadm-${KUBEEDGE_VERSION}-linux-${ARCH}

KIND_CLUSTER_NAME := kubeedge

.PHONY: download-kubeedge install-kubeedge local-cluster

install-keadm:
	mkdir -p tempdir
	wget \
		--directory-prefix ./tempdir \
		https://github.com/kubeedge/kubeedge/releases/download/${KUBEEDGE_VERSION}/${KEADM_FNAME}.tar.gz
	tar \
		--directory ./tempdir \
		-xvzf tempdir/${KEADM_FNAME}.tar.gz
	cp -r ./tempdir/${KEADM_FNAME}/* ./kubeedge
	rm -rf tempdir

logs-edgecore:
	journalctl -u edgecore.service -b -f

restart-edgecore:
	sudo service edgecore restart

delete-edgecore:
	sudo ./kubeedge/keadm/keadm reset

local-cluster:
	kind delete cluster \
		--name ${KIND_CLUSTER_NAME}
	kind create cluster \
		--config=config/kind.yaml \
		--name ${KIND_CLUSTER_NAME}

install-cloudcore:
	kubectl create ns kubeedge || true
	helm upgrade --install \
		kubeedge \
		./charts/kubeedge \
		--namespace kubeedge

cloudcore-token:
	kubectl get secret \
		-n kubeedge \
		tokensecret \
		-o=jsonpath='{.data.tokendata}' | base64 -d

