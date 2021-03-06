import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        Function to read in the two input csv files and return a dataframe consisting of the merge of them.
        
        INPUT:
        csv files:
            messages.csv
            categories.csv
        
        OUTPUT:
        Pandas dataframe:
            df - merge of messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, how='inner', on='id')
    
    return df


def clean_data(df):
    """
        Function to clean the duplicates in the merged dataframe and the expand the values in the column 'categories' and add a binary encoding.
        
        INPUT:
        Pandas dataframe:
            df - merge of messages and categoriesdf
        
        OUTPUT:
        Pandas dataframe:
            df - cleaned and encoded merge of messages and categories
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # extract a list of new column names for categories
    category_colnames = [i[:-2] for i in row]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert the occurences of the categories columns to integers 0 or 1 
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)     

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)   
    
    # check for categories having values different than 0 and 1. Only the related column contains a 2 which is assumed to be 1 and replaced accordingly
    {i: df[i].unique() for i in category_colnames}
    df['related'].replace(2, 1, inplace=True)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
        Function to export the cleaned dataframe to a SQL database.
        
        INPUT:
        Pandas dataframe:
            df - cleaned and encoded merge of messages and categories
        
        OUTPUT:
        SQL database:
            database_filename
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
