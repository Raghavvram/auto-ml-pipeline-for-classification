from automlplforidps import run_classification_pipeline

def main():
    my_file_path = '../data/0.01_dist_percent_8classes.csv'
    my_target_column = 'label'

    print("--- Starting Automated Classification Pipeline ---")
    
    trained_models = run_classification_pipeline(
        file_path=my_file_path,
        target_column=my_target_column
    )
    
    print("\n--- Pipeline Complete ---")
    
    if trained_models:
        print("\nAvailable trained models:", list(trained_models.keys()))
        
        rf_model = trained_models.get("RandomForest")
        if rf_model:
            print("\nSuccessfully retrieved the tuned RandomForest model for future use.")

main()
