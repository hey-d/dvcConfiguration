import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import os


nltk.download('punkt_tab', quiet=True) 


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def transform_text(text: str)->str:
    try:
        ps = PorterStemmer()
        text = text.lower()
        text = nltk.word_tokenize(text)
        text = [word for word in text if word.isalnum()]
        text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
        return ' '.join(text)
    except Exception as e:
        print(f"error transforming the text, {e}")
        return text
    
def preprocess_data(df: pd.DataFrame, text_data = 'v2', target_data = 'v1')->pd.DataFrame:
    try:
        encoder = LabelEncoder()
        df = df.drop_duplicates(keep='first')
        print("duplicates removed successfully")
        df[target_data] = encoder.fit_transform(df[target_data])
        print("target data encoded successfully")
        df[text_data] = df[text_data].apply(transform_text)
        print("text data transformed successfully")
        
        return df;
    except Exception as e:
        print(f"error prepocessing the data, {e}")
        return None
def main():
    
    train_data = pd.read_csv('./data/raw/train_data.csv')
    test_data = pd.read_csv('./data/raw/test_data.csv')
    print("data loaded successfully")
    
    preProcessed_train = preprocess_data(train_data)
    preProcessed_test = preprocess_data(test_data)
    print("data processed successfully")
    
    
    newPath = os.path.join('./data', 'interim')
    os.makedirs(newPath, exist_ok=True)
    print("interim directory created successfully")
    
    interim_train = os.path.join(newPath, 'new_train_data.csv')
    interim_test = os.path.join(newPath, "new_test_data.csv")
    print("interim file paths defined")
    
    preProcessed_train.to_csv(interim_train, index=False)
    preProcessed_test.to_csv(interim_test, index=False)
    print("preprocessed and new data saved successfully")
    
if __name__ == "__main__":
    main()