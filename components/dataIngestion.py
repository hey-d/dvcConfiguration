import pandas as pd
import os 
from sklearn.model_selection import train_test_split

def load_data(file_path: str)->pd.DataFrame:
    try: 
        df=pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"error loading the data, {e}")
        return None
    
def process_data(df: pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        print("data processed Successfullt")
        return df;
    except Exception as e:
        print(f"error processing the data, {e}")
        return None

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, path: str):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        train_data.to_csv(os.path.join(path, 'train_data.csv'), index=False)
        print("train data saved successfully")
        test_data.to_csv(os.path.join(path, 'test_data.csv'), index=False)
        print("test data saved successfully")
        
    except Exception as e:
        print(f"error saving the data, {e}")
        

def main():
    data_path = './raw/spam.csv'
    df= load_data(data_path)
    if df is not None:
        processed_df = process_data(df)
        if processed_df is not None:
            test_size = 0.2
            random_state=42
            train_data, test_data = train_test_split(processed_df, test_size = test_size, random_state = random_state)
            path = './data'
            save_data(train_data, test_data, path)
            
if __name__ == "__main__": main()

            