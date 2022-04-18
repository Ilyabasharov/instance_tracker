#!/bin/bash

export DOCKER_UID
export DOCKER_IMAGE_NAME
export DOCKER_IMAGE_VERSION
export DOCKER_USER_NAME
export DOCKER_USER_PASSWORD
export DOCKER_PROJECT_PATH
export DOCKER_PROJECT_NAME
export DOCKER_DEFAULT_PATH
export DOCKER_ROS_DISTRO

yellow=`tput setaf 3`
green=`tput setaf 2`
violet=`tput setaf 5`
reset_color=`tput sgr0`

ARCH="$(uname -m)"

main () {

    if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "aarch64" ]; then
        file="docker/Dockerfiles/Dockerfile.${ARCH}"
    else
        echo "There is no Dockerfile for ${yellow}${ARCH}${reset_color} arch"
    fi

    echo "Building image for ${yellow}${ARCH}${reset_color} arch. from Dockerfile: ${yellow}${file}${reset_color}"
    
    docker build . \
        --file ${file} \
        --tag ${ARCH}/${DOCKER_ROS_DISTRO}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION} \
        --build-arg UID=${DOCKER_UID} \
        --build-arg GID=${DOCKER_UID} \
        --build-arg PW=${DOCKER_USER_PASSWORD} \
        --build-arg USER=${DOCKER_USER_NAME} \
        --build-arg PROJECT_PATH=${DOCKER_PROJECT_PATH} \
        --build-arg PROJECT_NAME=${DOCKER_PROJECT_NAME} \
        --build-arg DEFAULT_PATH=${DOCKER_DEFAULT_PATH} \
        --build-arg ROS_DISTRO=${DOCKER_ROS_DISTRO}
}

main "$@"; exit;