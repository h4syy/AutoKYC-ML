# Use a suitable Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    python3-dev \
    libatlas-base-dev \
    liblapack-dev \
    gfortran \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

RUN pip install --no-cache-dir torch torchvision torchaudio ultralytics

# Copy the application code
COPY . .

# Command to run your application
CMD ["python", "main.py"]