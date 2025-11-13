FROM python:3.11-trixie

WORKDIR /var/baseimage

# Copy requirements
ARG REQUIREMENTS_FILE="../requirements/backend.txt"
COPY ${REQUIREMENTS_FILE} ./requirements.txt

# # Alpine: install build-time dependencies
# RUN apk --no-cache add musl-dev linux-headers g++

RUN apt-get update -y

# Install os dependency for h5py
RUN apt-get install -y libhdf5-dev

# # Upgrade pip
# RUN python3 -m pip install --upgrade pip

# Cython for building, numpy < 2 avoids ndarray size changed ValueError
RUN pip3 install Cython numpy==1.25
# Upgrade from wsts reqs h5py==3.7.0; build seems to fail on 3.7
RUN pip3 install --no-binary=h5py h5py==3.15
# Install other requirements from file
RUN pip3 install -r requirements.txt