import pandas as pd
import pickle
import statsmodels.api as sm

def predict_ols():
    # Load the data
    # data = pd.read_csv('../../data/customer_churn_dataset/activation_data.csv')
    data = pd.read_csv('/tmp/activationBase/activation_data.csv')

    
    # Drop the 'Exited' column if it exists
    if 'Exited' in data.columns:
        data = data.drop(columns=['Exited'])
    
    # print(data.info())

    # Load the trained OLS model
    # with open('../../learningBaseOLS/ols_model.pkl', 'rb') as file:
    with open('/tmp/knowledgeBase/currentOlsSolution.pkl', 'rb') as file:
        model = pickle.load(file)

    # Add a constant
    data = sm.add_constant(data, has_constant='add')

    # Make predictions
    predictions = model.predict(data)

    # Convert continuous output to binary (0 or 1) using a threshold of 0.5
    binary_predictions = (predictions >= 0.5).astype(int)

    return binary_predictions

if __name__ == "__main__":
    predictions = predict_ols()
    print("Prediction", predictions.values[0])
