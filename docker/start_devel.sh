#!/bin/bash

export DOCKER_IMAGE_VERSION
export DOCKER_IMAGE_NAME
export DOCKER_CONTAINER_NAME
export DOCKER_USER_NAME
export DOCKER_PROJECT_PATH
export DOCKER_PORT_FROM
export DOCKER_PORT_TO
export DOCKER_PROJECT_NAME
export DOCKER_DEFAULT_PATH
export DOCKER_ROS_DISTRO

cd "$(dirname "$0")"
cd ..

workspace_dir=$PWD
ARCH="$(uname -m)"

desktop_start() {
    docker run \
        -itd \
        --ipc host \
        --publish ${DOCKER_PORT_FROM}:${DOCKER_PORT_TO} \
        --gpus all \
        --name ${DOCKER_CONTAINER_NAME} \
        --net "host" \
        --env "DISPLAY" \
        --env "QT_X11_NO_MITSHM=1" \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume ${workspace_dir}:${DOCKER_DEFAULT_PATH}/${DOCKER_USER_NAME}/${DOCKER_PROJECT_PATH}/${DOCKER_PROJECT_NAME}:rw \
        --privileged \
        ${ARCH}/${DOCKER_ROS_DISTRO}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION}
}

arm_start() {
    docker run \
        -itd \
        --ipc host \
        --publish ${DOCKER_PORT_FROM}:${DOCKER_PORT_TO} \
        --runtime nvidia \
        --name ${DOCKER_CONTAINER_NAME} \
        --network host \
        --env "DISPLAY" \
        --env "QT_X11_NO_MITSHM=1" \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --volume ${workspace_dir}:${DOCKER_DEFAULT_PATH}/${DOCKER_USER_NAME}/${DOCKER_PROJECT_PATH}/${DOCKER_PROJECT_NAME}:rw \
        --privileged \
        ${ARCH}/${DOCKER_ROS_DISTRO}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_VERSION}
}

main () {

    if [ "$(docker ps -aq -f status=exited -f name=${DOCKER_CONTAINER_NAME})" ]; then
        docker rm --force ${DOCKER_CONTAINER_NAME};
    fi

    if [ "$ARCH" = "x86_64" ]; then 
        desktop_start;
    elif [ "$ARCH" = "aarch64" ]; then
        arm_start;
    fi

    docker exec \
        -it \
        --user ${DOCKER_USER_NAME} \
            ${DOCKER_CONTAINER_NAME} \
            bash -c "cd ${DOCKER_DEFAULT_PATH}/${DOCKER_USER_NAME}/${DOCKER_PROJECT_PATH}/${DOCKER_PROJECT_NAME}; bash"
}

main;
