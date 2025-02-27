# Use an official PyTorch Lightning runtime as a parent image
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /code
ENV DEBIAN_FRONTEND noninteractive
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt update && apt install git -y

# Add SSH keys from host to the container
RUN mkdir ~/.ssh && ln -s /run/secrets/host_ssh_key ~/.ssh/id_rsa

