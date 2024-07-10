# main.py
import utils
from preprocess_data import preprocess_data

def main():
    # Load data
    data = load_data()

    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Perform main logic
    result = perform_main_logic(preprocessed_data)

    # Save result
    save_result(result)

if __name__ == "__main__":
    main()