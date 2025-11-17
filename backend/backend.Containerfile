FROM wispr-backend-base

# copy backend/ into /app/backend
WORKDIR /app
COPY backend ./backend

# copy wsts source
ARG WSTS_ROOT_HOST=src/wsts
COPY ${WSTS_ROOT_HOST} ./wsts
ENV WSTS_ROOT=/app/wsts

# CMD ["python", "src/preprocess/CreateHDF5Dataset.py", "--data_dir", "$SOURCE_DIR", "--target_dir", "$TARGET_DIR"]