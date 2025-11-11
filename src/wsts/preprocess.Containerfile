FROM python:3.11-trixie

WORKDIR /script

COPY ./src/preprocess ./src/preprocess
COPY ./src/dataloader ./src/dataloader
COPY ./src/__init__.py ./src/__init__.py
COPY ./preprocess-requirements.txt ./requirements.txt

# alpine: install build-time dependencies
# RUN apk --no-cache add musl-dev linux-headers g++

RUN apt-get update -y
RUN apt-get install -y libhdf5-dev # os dependency for h5py

# upgrade pip
# RUN python3 -m pip install --upgrade pip

RUN pip3 install Cython numpy==1.25.0 # Cython for building, numpy < 2 avoids ndarray size changed ValueError
RUN pip3 install --no-binary=h5py h5py==3.15.1 # upgrade from wsts reqs h5py==3.7.0; build seems to fail on 3.7
RUN pip3 install -r requirements.txt # install other requirements needed for preprocess, minus numpy and h5py

# for build-and-run locally
# ARG SOURCE_DIR
# ARG TARGET_DIR

# CMD ["python", "src/preprocess/CreateHDF5Dataset.py", "--data_dir", "$SOURCE_DIR", "--target_dir", "$TARGET_DIR"]
