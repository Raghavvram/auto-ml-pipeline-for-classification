from automlplforidps import run_classification_pipeline
import pandas as pd
import numpy as np
import joblib

def run_training_pipeline():
    print("--- PART 1: TRAINING AND SAVING THE BEST MODEL ---")
    
    my_file_path = 'path/to/your/dataset.csv'
    my_target_column = 'your_target_label'

    try:
        trained_models = run_classification_pipeline(
            file_path=my_file_path,
            target_column=my_target_column
        )
        if trained_models:
            print("\nPipeline training complete. Artifacts have been saved.")
        else:
            print("\nPipeline training failed.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("Please ensure 'path/to/your/dataset.csv' is a valid file path.")

def run_prediction_on_new_data():
    print("\n\n--- PART 2: LOADING ARTIFACTS FOR A NEW PREDICTION ---")
    
    try:
        artifacts = joblib.load('classification_artifacts.joblib')
        model = artifacts['model']
        scaler = artifacts['scaler']
        label_encoder = artifacts['label_encoder']
        selected_features = artifacts['selected_features']
        
        print("Successfully loaded model, scaler, and label encoder.")
        print(f"Model expects {len(selected_features)} features.")
        
        # --- Create a new data sample for prediction ---
        # NOTE: The new data must have the same features that the model was trained on.
        # This sample data is randomly generated for demonstration purposes.
        # Replace these values with your actual new data.
        # The data should be pre-processed (imputed, numerically encoded) before scaling.
        
        new_sample_values = np.random.rand(1, len(selected_features))
        new_data = pd.DataFrame(new_sample_values, columns=selected_features)
        
        print("\nGenerated a new sample for prediction:")
        print(new_data)
        
        # 1. Scale the new data using the loaded scaler
        new_data_scaled = scaler.transform(new_data)
        
        # 2. Make a prediction
        prediction_encoded = model.predict(new_data_scaled)
        
        # 3. Decode the prediction to the original label
        prediction_label = label_encoder.inverse_transform(prediction_encoded)
        
        print(f"\n---> Model Prediction: '{prediction_label[0]}'")

    except FileNotFoundError:
        print("\nError: 'classification_artifacts.joblib' not found.")
        print("Please run the training pipeline first to generate the artifacts.")
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")

run_training_pipeline()
run_prediction_on_new_data()