FROM node:24-alpine

WORKDIR /app

# Copy package files
ARG PACKAGE_DIR="./frontend/wildfireFE/"
COPY ${PACKAGE_DIR}/package*.json ./

# Install dependencies
RUN npm install --no-audit

# Copy other files
COPY ${PACKAGE_DIR} .

# Run build for production
RUN npm run build