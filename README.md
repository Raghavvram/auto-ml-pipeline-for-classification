# AutoML Pipeline for Tabular Classification

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/automlplforidps.svg)](https://badge.fury.io/py/automlplforidps)

A robust, automated pipeline for multiclass classification tasks on tabular data, designed to streamline the machine learning workflow from data preprocessing to model tuning. This project is particularly well-suited for complex classification problems, such as those found in Intrusion Detection and Prevention Systems (IDPS).

## Features

- **End-to-End Automation:** Handles the entire ML pipeline: preprocessing, balancing, feature engineering, and hyperparameter optimization.
- **Robust Preprocessing:** Includes handling for missing values, infinite values, and categorical data encoding.
- **Advanced Data Balancing:** Utilizes a hybrid approach with BorderlineSMOTE for oversampling and TomekLinks for cleaning noisy data.
- **Intelligent Feature Selection:** Employs Particle Swarm Optimization (PSO) to identify the most impactful features, reducing model complexity and improving performance.
- **Sophisticated Hyperparameter Tuning:** Uses Bayesian Search Cross-Validation to efficiently find optimal hyperparameters for multiple models.
- **Multi-Model Evaluation:** Trains, tunes, and evaluates several powerful gradient boosting and ensemble models:
    - RandomForest
    - XGBoost
    - CatBoost

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/autoMLplforIDPS.git
cd autoMLplforIDPS
pip install -e .
```

This project uses Python 3.12+ and the dependencies are listed in `pyproject.toml`.

## Quick Start

Using the pipeline is straightforward. Simply provide the path to your dataset and specify the target column. The pipeline returns a dictionary of trained models.

See the example below (`example.py`):

```python
from automlplforidps import run_classification_pipeline

def main():
    # Path to your data and the name of the column to predict
    my_file_path = 'path/to/your/dataset.csv'
    my_target_column = 'your_target_label'

    print("--- Starting Automated Classification Pipeline ---")
    
    # Run the entire pipeline
    trained_models = run_classification_pipeline(
        file_path=my_file_path,
        target_column=my_target_column
    )
    
    print("
--- Pipeline Complete ---")
    
    if trained_models:
        print("
Available trained models:", list(trained_models.keys()))
        
        # You can now access the tuned models for inference
        best_rf_model = trained_models.get("RandomForest")
        if best_rf_model:
            print("
Successfully retrieved the tuned RandomForest model.")
            # Example: best_rf_model.predict(new_data)

if __name__ == "__main__":
    main()
```

## The Pipeline Stages

The pipeline executes the following steps in sequence:

1.  **Load & Preprocess Data:** Reads the CSV, handles missing/infinite values, and encodes labels.
2.  **Balance Data:** Corrects class imbalance using BorderlineSMOTE and TomekLinks.
3.  **Normalize Features:** Scales all numerical features using `StandardScaler`.
4.  **Feature Engineering (PSO):** Selects the optimal feature subset using Particle Swarm Optimization.
5.  **Tune & Evaluate:** Performs Bayesian hyperparameter optimization for each model and provides a detailed performance evaluation, including per-class accuracy.

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

trained_models = run_classification_pipeline(
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
