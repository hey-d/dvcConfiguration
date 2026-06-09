import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import yaml



def load_params(params_path: str)->dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            print("params loaded successfully")
            return params
    except Exception as e:
        print(f"error occured while loading params, {e}")
        return None

def load_data(file_path: str)->pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df= df.fillna('')
        print("data loaded and missing values handled successfully")
        return df
    except Exception as e:
        print(f"error loading the data, {e}")
        return None
    
def vectorize_text(train_data:pd.DataFrame, test_data:pd.DataFrame, limit: int)->tuple:
    try: 
        vectorizer = TfidfVectorizer(max_features=limit)
        
        X_train = train_data['v2'].values
        X_test = test_data['v2'].values
        y_train = train_data['v1'].values
        y_test = test_data['v1'].values
        
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        print("vectorization feature distribution completed")
        
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label']=y_train
        print("training vectorization completed")
        
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label']=y_test
        print("testing vectorization completed")
        
        return train_df, test_df
    except Exception as e:
        print(f"error vectorizing the text data, {e}")
        return None, None
        
        
def main():
    
    train_path = './data/interim/new_train_data.csv'
    test_path = './data/interim/new_test_data.csv'
    
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    if train_data is not None and test_data is not None:
        params = load_params('params.yaml')
        limit = params['feature-engineering']['limit']
        train_df, test_df = vectorize_text(train_data, test_data, limit)
        if train_df is not None and test_df is not None:
            newPath =  os.path.join('./data', 'processed')
            os.makedirs(newPath, exist_ok=True)
            print("processed directory created successfully")
            
            processed_train_path  = os.path.join(newPath, 'processed_train.csv')
            processed_test_path = os.path.join(newPath, 'processed_test.csv')
            print("processed file paths defined")
            
            train_df.to_csv(processed_train_path, index=False)
            test_df.to_csv(processed_test_path, index=False)
            print("feature engineering completed successfully")

if __name__ == "__main__":
    main()