import os
import sys
import json
from utils import preprocess_data, feature_engineering

def main():
    # Load project structure from JSON file
    with open('project_structure.json', 'r') as f:
        project_structure = json.load(f)

    # Preprocess data using utility functions
    preprocessed_data = preprocess_data(project_structure['data_path'])

    # Perform feature engineering
    engineered_features = feature_engineering(preprocessed_data)

    # Save engineered features to a file
    with open('engineered_features.json', 'w') as f:
        json.dump(engineered_features, f)

if __name__ == '__main__':
    main()