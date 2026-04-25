import pandas as pd
import joblib

from preprocess import preprocess_data


def predict():
    model = joblib.load("model/loan_model.pkl")
    scaler = joblib.load("model/scaler.pkl")

    test_df = pd.read_csv("data/test.csv")
    loan_ids = test_df['Loan_ID']

    X = preprocess_data(test_df)

    X = scaler.transform(X)

    preds = model.predict(X)

    output = pd.DataFrame({
        "Loan_ID": loan_ids,
        "Loan_Status": preds
    })

    output.to_csv("submission.csv", index=False)

    print("Predictions saved to submission.csv")


if __name__ == "__main__":
    predict()