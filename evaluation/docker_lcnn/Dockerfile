FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Arguments
ARG user=pytorch
ARG uid=1000
ARG gid=1000
ARG gpu_arch=sm_61

ENV DEBIAN_FRONTEND noninteractive

# Install some dependencies
RUN apt-get update && apt-get install -y \
		wget\
    git\
		python3-dev\
    python3-pip\
    sudo\
		vim\
		unzip\
	  && \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

#Install repo dependencies
RUN pip3 install --upgrade pip  &&\
		pip3 install --no-cache-dir\
		opencv-python-headless\
		torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html\
		tb-nightly\
		tensorboardX\
		future \
		Cython \
		matplotlib \
		numpy \
		scipy \
		pyyaml \
		gitpython \
		scikit-image\
		seaborn \
		tqdm \
		docopt

# Setup user
RUN export uid="${uid}" gid="${gid}" && \
    groupadd -g "${gid}" "${user}" && \
    useradd -m -u "${uid}" -g "${user}" -s /bin/bash "${user}" && \
    passwd -d "${user}" && \
    usermod -aG sudo "${user}"

# Switch to user
USER "${uid}"

VOLUME /data
VOLUME /host_home

# Default powerline10k theme, no plugins installed
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"

WORKDIR /host_home
CMD zsh
