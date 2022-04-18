# PointTrack

## Introduction

[Object tracking algorithm](https://github.com/Ilyabasharov/instance_tracker/docs/point_track.md) based on cloud of points using segmentation masks. It was improved in case of speed using GPU-based preprocessing and numba-based assinging algorithm.

This branch works with **PyTorch 1.10.0**.

<div align="center">
[💪Introduction](https://github.com/Ilyabasharov/instance_tracker/blob/master/README.md#introduction) |
[🛠️Installation](https://github.com/Ilyabasharov/instance_tracker/blob/master/README.md#installation) |
[🏃Run](https://github.com/Ilyabasharov/instance_tracker/blob/master/README.md#run) |
[👀Contents](https://github.com/Ilyabasharov/instance_tracker/blob/master/README.md#contents) |
[🔖Docs](https://github.com/Ilyabasharov/instance_tracker/tree/master/docs)

</div>

## Installation

### Requirements

```bash
git-lfs
docker
nvidia-jetpack-4.5 (L4T R32.5.0) #if you are using NVIDIA edge devices
```

### Environment installation

```bash
git clone --branch indexing_fast https://gitlab.com/sdbcs-nio3/itl_mipt/segm_tracking/alg/tracking/pointtrack.git
cd pointtrack
git lfs fetch && git lfs pull
source docker/docker_names.sh
sh docker/build.sh
sh docker/start_devel.sh
```
See additional information in [docs/docker.md](https://github.com/Ilyabasharov/instance_tracker/blob/master/docs/docker.md)

After that you will be in project directory. You will need to register the package manually using scripts below.

### ROS package installation

```bash
cd ..
git clone https://gitlab.com/sdbcs-nio3/itl_mipt/ros_common/camera_objects_msgs.git
cd ..
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```
### Train environment installation

```bash
pip install -r requirements/train.txt
```

## Run the project

### ROS node

```bash
roslaunch pointtrack main.launch \
    camera_ns:=/stereo/left \            # camera namespace
    image_topic:=image_rect \            # colored image topic 
    objects_topic:=objects \             # topic with detection results (see camera object msgs)
    objects_track_ids_topic:=track_ids \ # ouput topic name
    print_stats:=1 \                     # print or dont print stat params
    stats_rate:=20                       # how often will the information be printed
```

### To train/test a model

See additional information in [docs/point_track.md](https://github.com/Ilyabasharov/instance_tracker/blob/master/docs/point_track.md)

### Improvements

There were a lot of research on speedup vs quality of work. The original network structure was changed, accelerated using [torch2trt](https://nvidia-ai-iot.github.io/torch2trt/master/getting_started.html). The dependence of the inference speed on the size of segmentation masks is investigated. Also the model has also been accelerated using [Numba](https://numba.pydata.org) library. You can also set the maximum number of objects during tracking in order to accurately meet the allocated inference time.
All of this improvements explained in [docs/point_track_improvements.md](https://github.com/Ilyabasharov/instance_tracker/blob/master/docs/point_track_improvements.md)

## Contents

```
├── docker                          <- Docker scripts and env setup.
├── docs                            <- Markdown files which provides an additional information about package.
├── launch                          <- Launch file for package params in the ROS namespace.
├── requirements                    <- Directory with main requirements for train/infer stage.
├── scripts                         <- Scripts for configuration, downloading weights.
│    │
│    ...
│    ├── ros_node.py                <- Entrypoint ros node file (inference).
│    ├── train_tracker_with_val.py  <- Entrypoint model file (training).
│    └── test_tracking.py           <- Entrypoint model file (testing).
│
...
│
├── weights                         <- pth-like files for the model storing in git lfs.
├── README.md                       <- You are here.
├── package.xml                     <- Main info about the package for ROS.
├── config.yaml                     <- Main config for running the ROS node.
└── requirements.txt                <- Required libraries.
```


