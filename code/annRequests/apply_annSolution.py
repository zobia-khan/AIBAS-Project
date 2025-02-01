import pandas as pd
from tensorflow.keras.models import load_model

def predict():
    # Load the data
    data = pd.read_csv('../../data/customer_churn_dataset/activation_data.csv')
    # Drop the 'exited' column if it exists
    if 'Exited' in data.columns:
        data = data.drop(columns=['Exited'])
    print(data.info())   
    # Load the model
    model = load_model('../../learningBase/ann_model.h5')
    
    # Make predictions
    predictions = model.predict(data)
        # Convert probability to binary (0 or 1) using a threshold of 0.5
    binary_predictions = (predictions >= 0.5).astype(int)
    
    return binary_predictions

if __name__ == "__main__":
    predictions = predict()
    print(predictions)
