#!/bin/bash
USE_NVIDIA=1 IMAGE=${IMAGE-lcnn} ./../../libs/dockers/common/build.sh "$@"
