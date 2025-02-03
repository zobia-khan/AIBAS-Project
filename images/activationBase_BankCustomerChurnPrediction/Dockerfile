# Dockerfile for activationBase
FROM busybox

# Copy activation data into the image
COPY ../../data/customer_churn_dataset/activation_data.csv /activationBase/

# Add the README file
COPY images/activationBase_BankCustomerChurnPrediction/Readme.md /activationBase/

# Set working directory
WORKDIR /tmp/activationBase