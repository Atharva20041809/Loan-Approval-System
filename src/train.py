import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from preprocess import preprocess_data


def train():
    # load data
    df = pd.read_csv("data/train.csv")

    # split features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # preprocess
    X = preprocess_data(X)

    # ✅ save feature columns (IMPORTANT)
    feature_columns = X.columns
    joblib.dump(feature_columns, "model/features.pkl")

    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # save model + scaler
    joblib.dump(model, "model/loan_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    print("Training complete. Model, scaler, and features saved.")


if __name__ == "__main__":
    train()