# Inference on (the) KubeEdge

This repository holds the notebooks and examples that will be shown in the [`Inference on (the) KubeEdge` talk during the Open Source Summit EU 2020](https://osseu2020.sched.com/event/eCFe/inference-on-the-kubeedge-adrian-gonzalez-martin-seldon).

## Pre-requirements

Most examples require you to have a Kubernetes cluster with KubeEdge installed, as well as a edge device in-sync with the cloud side.
We've included some Helm charts and config files that should help you get started.

### Disclaimer :warning:

These instructions haven't been tested fully across multiple environments, so **they are provided as-is**!
If you run into any issues, you should check each project's relevant docs:

- [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [Helm](https://helm.sh/)
- [keadm](https://docs.kubeedge.io/en/latest/setup/keadm.html)

### Kubernetes cluster

You can use [Kind](https://github.com/kubernetes-sigs/kind) to run a local cluster.
We've included a config file that exposes the ports used by `cloudcore`.

Once you've got Kind installed, it should be enough to run:


```python
!kind create cluster \
		--config=./config/kind.yaml \
		--name kubeedge
```

### KubeEdge

We've included a simple [Helm](https://helm.sh/) chart that installs `cloudcore`, the cloud-side component of KubeEdge, exposing the ports `10000` and `10002` used to sync with the edge-side.
If you've got Helm installed, it should be enough to run:

> :warning: Make sure you are pointing to your local Kind cluster!


```python
!kubectl create ns kubeedge
!helm install \
		kubeedge \
		./charts/kubeedge \
		--namespace kubeedge
```

### Edge Device

Through our examples we've used a Raspberry Pi 3 Model B V1.2 with an ARM architecture of 32 bits.
This device will need to have the edge-side component of KubeEdge (`edgecore`) installed and running.

To do this, we can use the [`keadm` tool](https://docs.kubeedge.io/en/latest/setup/keadm.html), built by the KubeEdge project.
To install it an edge device you can download it directly from the [project's GitHub release page](https://github.com/kubeedge/kubeedge/releases/download/v1.4.0/keadm-v1.4.0-linux-arm.tar.gz).
You can also leverage the included `Makefile` target as:

> :warning: This needs to be run on the edge-side!


```python
!make install-keadm
```

Once downloaded, you will need to join the device to the cluster.
To do this, you first need a secret token from the cloud-side.
This token can be retrieved as:

> :warning: Note that this command needs to run on the cloud side!


```python
!kubectl get secret \
		-n kubeedge \
		tokensecret \
		-o=jsonpath='{.data.tokendata}' | base64 -d
```

Once `keadm` is installed and we've got the token, we can then join to the cloud side by running:

> :warning: This needs to be run on the edge-side!


```python
!sudo keadm join \
    --cloudcore-ipport=${CLOUD_SIDE_IP}:10000 \
    --token=${CLOUD_SECRET_TOKEN} \
    --interfacename ${INTERNET_INTERFACE:wlan0} \
    -i raspberry
```

## Usage

Once your environment is set up, you can check the examples shown during the talk:

- [LED Blinker](./examples/led)
- [Face Mask Detector](./examples/face-mask-detector)


```python

```
