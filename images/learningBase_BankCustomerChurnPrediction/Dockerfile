# Dockerfile for learningBase
FROM busybox

# Copy training and test data into the image
COPY ../../data/customer_churn_dataset/training_data.csv /learningBase/train/
COPY ../../data/customer_churn_dataset/test_data.csv /learningBase/validation/

# Add the README file
COPY images/learningBase_BankCustomerChurnPrediction/Readme.md /learningBase/

# Set working directory
WORKDIR /tmp/learningBase
