# Notes

## Cloud

- WIP Helm Chart to install cloud component
  - Need to expose both `10000` (websocket) and `10002` (cloudhub)
- Need to build `kubeedge/cloudcore:v1.4.0` manually
  ```
  cd kubeedge
  make cloudimage
  kind load docker-image
  ```
  - otherwise won't work with latest version of CRD
  - `make cloudimage` && `kind load docker-image`

## Edge

- Install `keadm` to join cluster (with arch `arm` or `arm32v7`)
- Need to edit `/boot/firmware/cmdline.txt`
  - Otherwise, error is:
    ```
    initialize module error: system validation failed - Following Cgroup subsystem not mounted: [memory]
    ```
  - From https://github.com/kubeedge/kubeedge/issues/1613#issuecomment-623289518

### Useful commands

- Show logs from `edgecore` service
  ```
  journalctl -u edgecore.service -b
  ```
- Restart `edgecore` service
  ```
  sudo service edgecore restart
  ```
