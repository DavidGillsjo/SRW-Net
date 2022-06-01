#!/bin/bash
#Usage: [ENV_OPTS] ./run_local [CMD] [ARGS]
USE_NVIDIA=1 IMAGE=${IMAGE-semantic-room-wireframe} ./../libs/dockers/common/run.sh "$@"
