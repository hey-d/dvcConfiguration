import pandas as pd
import os 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import json
from dvclive import Live
import yaml

def load_data(file_path: str)->pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print("file loaded successfully")
        return df
    except Exception as e:
        print(f"error loading the file, {e}")
        return None

def load_model(file_path: str)->pickle:
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            return model;
        
    except Exception as e:
        print(f"error loading the model, {e}")
        return None
    
def load_params(params_path: str)->dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            print("params loaded successfully")
            return params
    except Exception as e:
        print(f"error occured while loading params, {e}")
        return None

def evaluate_model(model:pickle, X_test: np.ndarray, y_test: np.ndarray)->dict:
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics={
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        return metrics
    except Exception as e:
        print(f"error evaluating the model, {e}")
        return None


def save_metrics(metrics: dict, file_oath:str):
    try:
        os.makedirs(os.path.dirname(file_oath), exist_ok=True)
        with open(file_oath, 'w') as file:
            json.dump(metrics, file)
    except Exception as e:
        print(f"error saving the metrics, {e}")
        
def main():
    try:
        
        params = load_params('params.yaml')
        clf = load_model('./models/random_forest_model.pkl')
        print("model loaded successfully")
        test_data = load_data('./data/processed/processed_test.csv')
        
        X_test = test_data.drop('label', axis=1).values
        y_test = test_data['label'].values
        
        metrics = evaluate_model(clf, X_test, y_test)
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', metrics['accuracy'])
            live.log_metric('precision', metrics['precision'])
            live.log_metric('recall', metrics['recall'])
            live.log_metric('f1', metrics['f1_score'])
            
            live.log_params(params)
        
        print("model evaluated successfully")
        
        if metrics is not None:
            save_metrics(metrics, './reports/metrics.json')
            print("metrics saved successfully")
    except Exception as e:
        print(f"error in the main function, {e}")
        
if __name__ == "__main__":
    main()

        
        