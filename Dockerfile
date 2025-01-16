FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

RUN apt-get update && apt-get install -y
build-essential
cmake
autoconf
automake
libtool
curl
g++
unzip
git
vim

RUN pip install --upgrade pip

RUN git clone https://github.com/LLNL/LEAP.git /root/LEAP
WORKDIR /root/LEAP
RUN pip install .
RUN pip install imageio
RUN pip install matplotlib
