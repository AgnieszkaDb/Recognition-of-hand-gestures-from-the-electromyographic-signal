import numpy as np
import json
from model_tools import evaluation

def main():
    # Load the original y_test, y_pred, and r_test_segments
    history_file = "history_hilbert_full.json"
    r_test_file = "r_test_segments.npy"
    
    with open(history_file, "r") as f:
        history_data = json.load(f)

    y_pred = np.array(history_data["y_pred"])
    y_test = np.array(history_data["y_test"])
    r_test_segments = np.load(r_test_file)
    print(r_test_segments.shape)
    print(y_test.shape)
    print(y_pred.shape)

    # Perform voting using the provided function
    _, _, test, pred = evaluation.evaluate_vote(y_test, y_pred, r_test_segments)
    print(test, pred)

    # Save the new y_pred and y_test
    np.save("y_pred_voted.npy", pred)  # Replace with the correct variable
    np.save("y_test_voted.npy", test)  # Replace with the correct variable

if __name__ == "__main__":
    main()
