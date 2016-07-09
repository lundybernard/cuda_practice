# cuda_practice

## Experiments:
### Get a docker image that we can run deepdream on

1. Caffe's provided base image
    - [caffe/gpu] (https://github.com/BVLC/caffe/blob/master/docker/standalone/gpu/Dockerfile)
    - test it: `docker run --rm -it caffe-cuda8.0-cudnn5-ubuntu16.04 caffe --version` >> caffe version 1.0.0-rc3
