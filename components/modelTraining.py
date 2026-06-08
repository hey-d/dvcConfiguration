import pandas as pd;
import os;
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


def load_data(file_path: str)->pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print("data loaded successfully")
        return df
    except Exception as e:
        print(f"error loading the data, {e}")
        return None

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict)->RandomForestClassifier:
    try:
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must be the same.")
        model = RandomForestClassifier(n_estimators = params['n_estimators'], random_state=params['random_state'])
        clf = model.fit(X_train, y_train)
        return clf;
        
    except ValueError as ve:
        print(f"Value error: {ve}")
        return None
    except Exception as e:
        print(f"error training the model, {e}")
        return None
    
def save_model(model: RandomForestClassifier, file_path: str):
    try: 
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
           pickle.dump(model, file)
    except Exception as e:
        print(f"error saving the model, {e}")

def main():
    params={'n_estimators': 3, 'random_state': 42}
    train_path = './data/processed/processed_train.csv'
    train_data = load_data(train_path)
    print("Data loaded")
    if train_data is not None:
       X_train = train_data.drop('label', axis=1).values
       y_train = train_data['label'].values
       model = train_model(X_train, y_train, params)
       print("model trained successfully")
       if model is not None:
           model_path ='./models/random_forest_model.pkl'
           save_model(model, model_path)
           print("model saved successfully")

if __name__ == "__main__":
    main()
      
    
        