FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN sed -i 's/# deb-src/deb-src/' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
        build-essential \
        git \
        python3-dev \
        python3-pip \
        python3-matplotlib \
        pkg-config \
        netcat \
        zsh && \
    apt-get build-dep -y python3-matplotlib

RUN pip3 install cython

ARG UNAME=hendrik
ARG UID=20022
ARG GID=100

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/zsh $UNAME

ARG BASE=/app
RUN mkdir -p ${BASE}
WORKDIR ${BASE}

RUN pip3 install cupy-cuda111
COPY requirements_docker.txt requirements.txt
RUN pip3 install -r requirements.txt

ARG WAIT_DIR=/opt/wait-for
RUN git clone https://github.com/eficode/wait-for.git ${WAIT_DIR}
RUN chmod +x ${WAIT_DIR}/wait-for

USER $UNAME
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
RUN cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

CMD ["/bin/zsh"]

