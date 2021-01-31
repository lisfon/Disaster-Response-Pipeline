import sys
import pandas as pd
import numpy as np
import nltk
import re
import pickle
from sqlalchemy import create_engine
nltk.download(['wordnet', 'punkt'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
        Function to read in the SQL database generated in the the ETL data processing 
        and consisting of a cleaned merge of the categories and messages datasets and
        to isolate the X and y variables.
        
        INPUT:
        SQL database:
            DisasterResponse.db
        
        OUTPUT:
        Pandas dataframes:
            X = messages df
            Y = categories df
            categories_name = columns of df
    """  
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)   

    # variables definition
    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns.values
    
    return X,Y,category_names

def tokenize(text):
    """
        Function to tokenize all text content from messages.
        
        INPUT:
        Pandas dataframes:
            DisasterResponse.db
        
        OUTPUT:
        None
            
    """  

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    

def build_model():
    """
        Function to build and tune the machine learning model with most efficient
        parameters identified with grid search.
        
        INPUT:
        Function:
            tokenize
        
        OUTPUT:
        Tuned model with classifier AdaBoost Algorithm
            
    """  
    
    # machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    
    # definition of tuning parameter
    parameters = {'clf__estimator__n_estimators': [25,100],
              'clf__estimator__learning_rate': [0.02, 2.0]
                }
    
    # grid search for better parameter
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=None, n_jobs=-1, verbose=3)

    return cv    

def evaluate_model(model, X_test, Y_test, category_names):
    """
        Function to train and test the tuned model and evaluate its performance.
        
        INPUT:
        Tuned model with classifier AdaBoost Algorithm
        Dataframe splitted for training and testing (X_test, Y_test)
        
        OUTPUT:
        Report of model evaluation
            
    """      
    
    # predict the dependent variable y with the tuned model
    y_pred_tuned = model.predict(X_test)
    
    # evaluate the quality of prediction with a classification report
    for i,name in enumerate(Y_test):
        print(name)
        report = classification_report(Y_test[name], y_pred_tuned[:, i])
    
    return report

def save_model(model, model_filepath):
    """
        Save model as pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()