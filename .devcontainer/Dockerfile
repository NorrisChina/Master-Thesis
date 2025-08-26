# 基础镜像：Ubuntu 24.04
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 安装 Python 3.12 + venv + pip
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /workspace

# 创建虚拟环境
RUN python3.12 -m venv .venv && \
    . .venv/bin/activate && \
    python -m pip install --upgrade pip

CMD ["/bin/bash"]