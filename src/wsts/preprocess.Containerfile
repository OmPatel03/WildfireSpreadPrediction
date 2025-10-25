# FROM python:3.11-alpine
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

WORKDIR /script

COPY ./src/preprocess ./src/preprocess
COPY ./src/dataloader ./src/dataloader
COPY ./src/__init__.py ./src/__init__.py
COPY ./requirements.txt ./requirements.txt

# alpine: install build-time dependencies
# RUN apk --no-cache add musl-dev linux-headers g++

# upgrade pip
# RUN python3 -m pip install --upgrade pip

RUN pip3 install -r requirements.txt

ARG SOURCE_DIR
ARG TARGET_DIR

CMD ["python", "src/preprocess/CreateHDF5Dataset.py", "--data_dir", "$SOURCE_DIR", "--target_dir", "$TARGET_DIR"]
