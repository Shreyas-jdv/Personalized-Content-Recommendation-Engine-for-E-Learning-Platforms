import numpy as np
from model_training import train_model
import pickle


def save_model(model, file_name="recommendation_model.pkl"):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)


def load_model(file_name="recommendation_model.pkl"):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def recommend(user_data):
    model = load_model()
    user_data = np.array(user_data).reshape(1, -1)
    recommendation = model.predict(user_data)

    return recommendation


if __name__ == "__main__":
    # Assuming the model is already trained and saved
    user_data = [0, 1, 2, 3, 4]  # Replace with actual user features
    recommendation = recommend(user_data)
    print(f"Recommended Content ID: {recommendation[0]}")
