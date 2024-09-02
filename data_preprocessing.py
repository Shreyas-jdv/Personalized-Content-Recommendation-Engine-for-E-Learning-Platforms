import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Example preprocessing steps
    data.fillna(0, inplace=True)  # Handling missing values
    features = data.drop(columns=['CourseName'])
    target = data['CourseName']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
