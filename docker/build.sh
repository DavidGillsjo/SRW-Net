#!/bin/bash
USE_NVIDIA=1 IMAGE=${IMAGE-semantic-room-wireframe} ./../libs/dockers/common/build.sh "$@"
