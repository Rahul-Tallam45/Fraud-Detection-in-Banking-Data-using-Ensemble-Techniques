import joblib
import pandas as pd
import numpy as np

def predict_fraudulence(input_data):
    # Load the saved model
    loaded_model = joblib.load('C:/Users/tallamrahul/Desktop/code folder/Extension/model_new.sav')
    
    # Prepare input data as DataFrame
    # input_df = pd.DataFrame([input_data], columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6'])
    input_array = np.array(input_data).reshape(1, -1)
    
    # Make prediction
    prediction = loaded_model.predict(input_array)
    
    # Return prediction result
    return "Fraudulent Transaction Detected" if prediction[0] == 1 else "Non-fraudulent Transaction Detected"

if __name__ == "__main__":
    # Take user input
    input_values = []
    for i in range(6):
        value = float(input(f"Enter value {i+1}: "))
        input_values.append(value)
    
    # Predict fraudulence
    result = predict_fraudulence(input_values)
    
    # Output prediction result
    print("Prediction Result:", result)
