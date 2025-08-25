# AutoML Pipeline for Tabular Classification

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/automlplforidps.svg)](https://badge.fury.io/py/automlplforidps)

A robust, automated pipeline for multiclass classification tasks on tabular data, designed to streamline the machine learning workflow from data preprocessing to model deployment. This project handles the entire lifecycle, from training and tuning to saving artifacts for future predictions. It is particularly well-suited for complex classification problems, such as those found in Intrusion Detection and Prevention Systems (IDPS).

## Features

- **End-to-End Automation:** Handles the entire ML pipeline: preprocessing, balancing, feature engineering, and hyperparameter optimization.
- **Robust Preprocessing:** Includes handling for missing values, infinite values, and categorical data encoding.
- **Advanced Data Balancing:** Utilizes a hybrid approach with BorderlineSMOTE for oversampling and TomekLinks for cleaning noisy data.
- **Intelligent Feature Selection:** Employs Particle Swarm Optimization (PSO) to identify the most impactful features.
- **Sophisticated Hyperparameter Tuning:** Uses Bayesian Search Cross-Validation to efficiently find optimal hyperparameters for multiple models.
- **Multi-Model Evaluation:** Trains and evaluates several powerful models (RandomForest, XGBoost, CatBoost) and reports overall and per-class accuracy.
- **Automatic Artifact Saving:** Automatically saves the best-performing model, scaler, label encoder, and feature list to a single file for easy deployment.

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/autoMLplforIDPS.git
cd autoMLplforIDPS
pip install -e .
```

This project uses Python 3.12+ and the dependencies are listed in `pyproject.toml`.

## Quick Start

Using the pipeline involves two main stages: training and prediction. The pipeline first trains multiple models, selects the best one, and saves the necessary components (artifacts) to a file. Afterwards, you can load these artifacts to make predictions on new, unseen data.

See the example below (`example.py`):

### Part 1: Training the Pipeline and Saving Artifacts

First, run the pipeline with your training dataset. This will find the best model and save it, along with other necessary objects, to `classification_artifacts.joblib`.

```python
from automlplforidps import run_classification_pipeline

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
            print("
Pipeline training complete. Artifacts have been saved.")
        else:
            print("
Pipeline training failed.")
    except Exception as e:
        print(f"
An error occurred during training: {e}")
        print("Please ensure 'path/to/your/dataset.csv' is a valid file path.")

run_training_pipeline()
```

### Part 2: Loading Artifacts and Making Predictions

Once the training is complete, you can load the saved artifacts to perform predictions on new data.

```python
import pandas as pd
import numpy as np
import joblib

def run_prediction_on_new_data():
    print("

--- PART 2: LOADING ARTIFACTS FOR A NEW PREDICTION ---")
    
    try:
        # Load the dictionary of artifacts saved during training
        artifacts = joblib.load('classification_artifacts.joblib')
        model = artifacts['model']
        scaler = artifacts['scaler']
        label_encoder = artifacts['label_encoder']
        selected_features = artifacts['selected_features']
        
        print("Successfully loaded model and artifacts.")
        
        # Create a new data sample for prediction
        # NOTE: The new data must have the same feature columns the model was trained on.
        new_sample_values = np.random.rand(1, len(selected_features))
        new_data = pd.DataFrame(new_sample_values, columns=selected_features)
        
        # 1. Scale the new data using the loaded scaler
        new_data_scaled = scaler.transform(new_data)
        
        # 2. Make a prediction
        prediction_encoded = model.predict(new_data_scaled)
        
        # 3. Decode the prediction to the original label
        prediction_label = label_encoder.inverse_transform(prediction_encoded)
        
        print(f"
---> Model Prediction: '{prediction_label[0]}'")

    except FileNotFoundError:
        print("
Error: 'classification_artifacts.joblib' not found.")
        print("Please run the training pipeline first to generate the artifacts.")

run_prediction_on_new_data()
```

## The Pipeline Stages

The pipeline executes the following steps in sequence:

1.  **Load & Preprocess Data:** Reads the CSV, handles missing/infinite values, and encodes labels.
2.  **Balance Data:** Corrects class imbalance using BorderlineSMOTE and TomekLinks.
3.  **Normalize Features:** Scales all numerical features using `StandardScaler`.
4.  **Feature Engineering (PSO):** Selects the optimal feature subset using Particle Swarm Optimization.
5.  **Tune & Evaluate:** Performs Bayesian hyperparameter optimization for each model and provides a detailed performance evaluation.
6.  **Save Artifacts:** The best model, scaler, label encoder, and selected features are saved to `classification_artifacts.joblib`.

## Customization

The pipeline can be customized by passing a configuration dictionary to the `run_classification_pipeline` function. You can adjust parameters for PSO, Bayesian optimization, and cross-validation.

```python
from automlplforidps import run_classification_pipeline

# Example of custom configuration
custom_config = {
    "random_state": 101,
    "train_size": 0.75,
    "pso_iterations": 15,
    "pso_population_size": 25,
    "bayes_iterations": 30,
    "cv_folds": 5,
}

run_classification_pipeline(
    file_path='path/to/your/dataset.csv',
    target_column='your_target_label',
    config=custom_config
)
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential changes or additions.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.