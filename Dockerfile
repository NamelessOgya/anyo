FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry>=1.7"

WORKDIR /workspace
