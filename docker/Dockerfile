FROM nvidia/cudagl:10.2-devel-ubuntu18.04

# Arguments
ARG user=pytorch
ARG uid=1000
ARG gid=1000
ARG gpu_arch=sm_70

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
		libgtk2.0\
		libglu1-mesa-dev\
		freeglut3-dev\
		mesa-common-dev\
		libosmesa6-dev\
		libegl1-mesa-dev\
		libglvnd-dev\
		libgl1-mesa-dev\
		libgles2-mesa-dev\
		# ----- PyMesh -----
		libeigen3-dev \
    libgmp-dev \
    libgmpxx4ldbl \
    libmpfr-dev \
    libboost-dev \
    libboost-thread-dev \
    libtbb-dev \
		# -----Open3D ---------
		libusb-1.0-0 &&\
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

#CMake for PyMesh
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.2/cmake-3.19.2-Linux-x86_64.sh && \
	bash cmake-3.19.2-Linux-x86_64.sh --skip-license

#GLVND setup for EGL support
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

#Install repo dependencies
RUN pip3 install --upgrade pip &&\
		pip3 install --no-cache-dir\
		opencv-python-headless\
		torch \
		torchvision \
		tb-nightly \
		future \
		Cython \
		matplotlib \
		numpy \
		scipy \
		pyyaml \
		gitpython \
		seaborn \
		pycairo\
		# Structured 3D
		open3d \
		descartes \
		shapely \
		panda3d\
		# HAWP
		yacs\
		requests\
		scikit-image\
		tabulate\
		networkx\
		tqdm \
		wandb \
		h5py

RUN pip3 install --no-cache-dir \
	  torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.1+cu102.html

# Build Pymesh
RUn git clone https://github.com/PyMesh/PyMesh.git && \
	cd PyMesh && \
	git submodule update --init

RUN cd PyMesh && \
	export PYMESH_PATH=`pwd` && \
	python3 setup.py build && \
	python3 setup.py install

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
