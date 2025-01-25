# Dockerfile for learningBase
FROM busybox

# Copy training and test data into the image
COPY ../../data/customer_churn_dataset/training_data.csv /tmp/learningBase/train/
COPY ../../data/customer_churn_dataset/test_data.csv /tmp/learningBase/validation/

# Add the README file
COPY images/learningBase_BankCustomerChurnPrediction/Readme.md /tmp/learningBase/

# Set working directory
#WORKDIR /tmp/learningBase
