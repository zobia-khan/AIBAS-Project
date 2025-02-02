# Dockerfile for knowledgeBase
FROM busybox

# Copy models into the image
COPY ../../learningBase/ann_model.h5 /tmp/knowledgeBase/currentAiSolution.h5
COPY ../../learningBaseOLS/ols_model.pkl /tmp/knowledgeBase/currentOlsSolution.pkl

# Add the README file
COPY images/knowledgeBase_BankCustomerChurnPrediction/Readme.md /tmp/knowledgeBase/

# Set working directory
#WORKDIR /tmp/knowledgeBase