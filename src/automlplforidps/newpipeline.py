import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks

import xgboost as xgb

from zoofs import ParticleSwarmOptimization

DEFAULT_CONFIG = {
    "random_state": 42,
    "train_size": 0.8,
    "pso_iterations": 10,
    "pso_population_size": 20,
}

def load_and_preprocess_data(file_path, target_column):
    print("1. Loading and preprocessing data...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None

    df.columns = [col.lower() for col in df.columns]
    target_column = target_column.lower()
    
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        return None, None, None
        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    X = df.drop(columns=target_column)
    y = df[target_column]

    for col in X.select_dtypes(include='bool').columns:
        X[col] = X[col].astype(int)

    numeric_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(include='object').columns

    X[numeric_cols] = SimpleImputer(strategy='median').fit_transform(X[numeric_cols])
    X[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[categorical_cols])

    le_features = LabelEncoder()
    for col in categorical_cols:
        X[col] = le_features.fit_transform(X[col].astype(str))
    
    le_target = LabelEncoder()
    if y.dtype == 'object':
        y = pd.Series(le_target.fit_transform(y), name=target_column)

    return X, y, le_target

def balance_data(X, y, config):
    print("2. Balancing data...")
    try:
        smote = BorderlineSMOTE(kind='borderline-1', random_state=config["random_state"])
        X_res, y_res = smote.fit_resample(X, y)
        tomek = TomekLinks()
        X_final, y_final = tomek.fit_resample(X_res, y_res)
        return pd.DataFrame(X_final, columns=X.columns), pd.Series(y_final, name=y.name)
    except Exception as e:
        print(f"Warning: Could not balance data. Proceeding with original data. Reason: {e}")
        return X, y

def normalize_features(X):
    print("3. Normalizing features...")
    if X.empty:
        return X, None
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, scaler

def engineer_features_pso(X, y, config):
    print("4. Engineering features with PSO...")
    if X.shape[1] < 5:
        print("Skipping PSO for dataset with less than 5 features.")
        return X

    def objective_function(model, X_train, y_train, X_valid, y_valid):
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)
        return log_loss(y_valid, proba)

    X_train_pso, X_test_pso, y_train_pso, y_test_pso = train_test_split(X, y, test_size=0.2, random_state=config["random_state"])
    
    pso_model = RandomForestClassifier(random_state=config["random_state"], n_jobs=-1)
    pso = ParticleSwarmOptimization(
        objective_function, n_iteration=config["pso_iterations"],
        population_size=config["pso_population_size"], minimize=True
    )
    pso.fit(pso_model, X_train_pso, y_train_pso, X_test_pso, y_test_pso, verbose=False)
    
    pso_features = pso.best_feature_list
    if not pso_features:
        print("Warning: PSO did not select any features. Using all features.")
        return X
        
    print(f"PSO selected {len(pso_features)} features.")
    return X[pso_features]

def get_models(num_classes, random_state):
    return {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "XGBoost": xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes, random_state=random_state, eval_metric='mlogloss'),
    }

def train_and_evaluate_model(model_name, params, X_train, y_train, X_test, y_test, num_classes, label_encoder, config):
    print(f"\n--- Training and Evaluating: {model_name} ---")
    
    model = get_models(num_classes, config["random_state"]).get(model_name)
    if model is None:
        print(f"Warning: Model '{model_name}' not found. Skipping.")
        return None, 0.0

    print(f"\n*** Using Predefined Hyperparameters for {model_name} ***")
    model.set_params(**params)
    sorted_params = OrderedDict(sorted(params.items()))
    for k, v in sorted_params.items(): print(f"  {k}: {v}")

    model.fit(X_train, y_train)

    print(f"\n--- Final Evaluation for {model_name} ---")
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Overall Accuracy (Test): {test_accuracy * 100:.2f}%")
    print(f"Overall Accuracy (Train): {accuracy_score(y_train, y_pred_train) * 100:.2f}%")
    
    print("\nIndividual Class Accuracy (Recall) on Test Set:")
    cm_test = confusion_matrix(y_test, y_pred_test, labels=label_encoder.transform(label_encoder.classes_))
    for i, class_name in enumerate(label_encoder.classes_):
        recall = cm_test[i, i] / (cm_test[i, :].sum() + 1e-9)
        print(f"  - Class '{class_name}': {recall * 100:.2f}%")

    return model, test_accuracy

def run_classification_pipeline(file_path, target_column, predefined_params, config=DEFAULT_CONFIG):
    X, y, label_encoder = load_and_preprocess_data(file_path, target_column)
    if X is None: return {}
        
    num_classes = y.nunique()
    print(f"Detected {num_classes} classes in the target variable.")

    X_bal, y_bal = balance_data(X, y, config)
    X_norm, scaler = normalize_features(X_bal)
    X_eng = engineer_features_pso(X_norm, y_bal, config)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y_bal, train_size=config["train_size"], random_state=config["random_state"]
    )
    
    if not predefined_params:
        print("Error: No predefined hyperparameters provided. Aborting.")
        return {}

    print("\n5. Starting Model Training and Evaluation with Predefined Hyperparameters...")
    trained_models = {}
    
    best_model_so_far = None
    best_accuracy = -1.0
    best_model_name = ""

    for name, params in predefined_params.items():
        trained_model, accuracy = train_and_evaluate_model(name, params, X_train, y_train, X_test, y_test, num_classes, label_encoder, config)
        if trained_model:
            trained_models[name] = trained_model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_so_far = trained_model
                best_model_name = name
    
    if best_model_so_far:
        print(f"\n--- Best Model ---")
        print(f"Best model found: {best_model_name} with {best_accuracy * 100:.2f}% accuracy.")

    return trained_models

# --- Example Usage ---
#
# # 1. Define the path to your data and the target column name.
# file = 'path/to/your/dataset.csv'
# target = 'YourTargetColumn'
#
# # 2. Define the hyperparameters you found during tuning.
# my_hyperparameters = {
#     "RandomForest": {
#         'n_estimators': 450,
#         'max_depth': 25
#     },
#     "XGBoost": {
#         'n_estimators': 600,
#         'max_depth': 8,
#         'learning_rate': 0.05
#     }
# }
#
# # 3. Run the pipeline with your predefined parameters.
# trained_models_dict = run_classification_pipeline(
#     file_path=file,
#     target_column=target,
#     predefined_params=my_hyperparameters
# )
