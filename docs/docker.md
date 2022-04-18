# docker

## Docker default values

In [docker/docker_names.sh](https://gitlab.com/sdbcs-nio3/itl_mipt/segm_tracking/alg/tracking/pointtrack/-/blob/indexing_fast/docker/docker_names.sh) you can see all default params for building container. For example,

```bash
# default password
DOCKER_USER_PASSWORD=user
# default username
DOCKER_USER_NAME=user
```
etc.

## Docker commands

If you just downloaded the project,

```bash
cd pointtrack
source docker/docker_names.sh
sh docker/build.sh
sh docker/start_devel.sh
```

If you want to enter the installed environment, 

```bash
cd pointtrack
source docker/docker_names.sh
sh docker/into.sh
```
If you want to delete the environment, 

```bash
cd pointtrack
source docker/docker_names.sh
sh docker/delete.sh
```
