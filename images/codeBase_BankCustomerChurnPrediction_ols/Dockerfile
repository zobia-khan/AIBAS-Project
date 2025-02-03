# Use the official Python 3.11 base image
FROM python:3.11-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default version
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install required Python packages
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==1.26.4\
    scikit-learn==1.6.1 \
    matplotlib==3.10.0 \
    seaborn==0.13.2\
    statsmodels==0.14.4\
    scipy==1.12.0

# Copy application code
COPY ../../code/olsRequests /codeBase/

# Add the README file
COPY images/codeBase_BankCustomerChurnPrediction_ols/Readme.md /codeBase/

# Set working directory
WORKDIR /tmp/codeBase/
