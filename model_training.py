import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



def train_model(file_path):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return model


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Convert date to numeric timestamp
    data['SignUpDate'] = pd.to_datetime(data['SignUpDate']).astype(
        int) / 10 ** 9  # Replace 'date_column' with the actual date column name

    data.fillna(0, inplace=True)

    label_encoder = LabelEncoder()
    data['UserID'] = label_encoder.fit_transform(data['UserID'])  # Replace 'user_id' with your column name

    features = data.drop(columns=['CourseName'])  # Replace 'target_column' with your target column
    target = data['CourseName']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model = train_model("C:/Users/Lenovo/PycharmProjects/Personalized Content Recommendation Engine for E-Learning Platforms/e_learning_dataset_with_course_names.csv")


