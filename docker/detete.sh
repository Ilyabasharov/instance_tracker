#!/bin/bash

export DOCKER_IMAGE_VERSION
export DOCKER_IMAGE_NAME
export DOCKER_CONTAINER_NAME
export DOCKER_ROS_DISTRO

ARCH="$(uname -m)"

main() {
    docker rm --force ${DOCKER_CONTAINER_NAME};
    docker image rm ${ARCH}/${DOCKER_ROS_DISTRO}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION};
}

main "$@"; exit;