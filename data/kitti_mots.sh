#!/bin/bash

MAIN_DIR=kitti_mots
mkdir -p ${MAIN_DIR}

IMAGES=https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip

wget --quiet \
    --show-progress \
    --progress=bar:force:noscroll \
    --no-check-certificate \
    ${IMAGES} \
    -O ${MAIN_DIR}/data_tracking_image_2.zip

unzip ${MAIN_DIR}/data_tracking_image_2.zip -d ${MAIN_DIR}
rm ${MAIN_DIR}/data_tracking_image_2.zip

ANNOTATIONS=https://www.vision.rwth-aachen.de/media/resource_files/instances.zip

wget --quiet \
    --show-progress \
    --progress=bar:force:noscroll \
    --no-check-certificate \
    ${ANNOTATIONS} \
    -O ${MAIN_DIR}/instances.zip

unzip ${MAIN_DIR}/instances.zip -d ${MAIN_DIR}
rm ${MAIN_DIR}/data_tracking_image_2.zip