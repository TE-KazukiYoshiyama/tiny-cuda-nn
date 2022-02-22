FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    libssl-dev \
    openssl

WORKDIR /opt

RUN echo "Installing cmake ..." \
    && wget https://github.com/Kitware/CMake/releases/download/v3.21.5/cmake-3.21.5.tar.gz \
    && tar zxvf cmake-3.21.5.tar.gz \
    && cd cmake-3.21.5 \
    && ./bootstrap \
    && make \
    && make install

RUN echo "Building tiny-cuda-nn ..." \
    && git clone --recursive https://github.com/nvlabs/tiny-cuda-nn \
    && cd tiny-cuda-nn \
    && cmake . -B build \
    && cmake --build build --config RelWithDebInfo -j 16

RUN echo "Installing PyTorch" \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip \
    && pip3 install ipython \
    && pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


# torch.cuda.is_available fails with docker build...
#RUN echo "Installing PyTorch extension ..." \
#    && cd tiny-cuda-nn/bindings/torch \
#    && python3 setup.py install


WORKDIR /opt
CMD [ "/bin/bash" ]    
    


