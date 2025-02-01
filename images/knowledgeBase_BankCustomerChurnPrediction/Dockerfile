# Dockerfile for knowledgeBase
FROM busybox

# Copy activation data into the image
COPY ../../learningBase/ann_model.h5 /tmp/knowledgeBase/currentAiSolution.h5
COPY ../../learningBaseOLS/ols_model.h5 /tmp/knowledgeBase/currentOlsSolution.h5

# Add the README file
COPY images/knowledgeBase_BankCustomerChurnPrediction/Readme.md /tmp/knowledgeBase/

# Set working directory
#WORKDIR /tmp/knowledgeBase