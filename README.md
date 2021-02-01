

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions and Instructions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run with no issues using Python versions 3.* Further libraries such as Pandas, Numpy, sklearn, nltk, SQL Alchemy and Pickle are required and their import will be automatically triggered withing the run of the data/process_data.py and models/train_classifier.py scripts.

## Project Motivation<a name="motivation"></a>

With this project, which is part of the Data Scientist Nanodegree Program of Udacity, I analysed the disaster data provided by Figure Eight to build a model for an API that classifies disaster messages.

## File Descriptions and results <a name="files"></a>

The repository is structured as follows:

- data: containing the two datasets used as input as well as the cleaned and prepared dataset obtained from their merge and the Python code for the data preparation. The detailed process can be viewed in the ETL Pipeline Preparation Jupyter notebook, while the final script to be run is saved as process_data.py.
- models: containing the machine learning pipeline to categorize the disaster events. The detailed process of evaluation and selection of the most suitable machine learning algorithm can be viewed in the ML Pipeline Preparation Jupyter notebook, while the final script with the tuned model is saved as train_classifier.py.
- app: containing the script for a web app showing visualizations of the data and a classification function

Instructions for running the scripts and using the web app:

- ETL pipeline: to clean data and store the processed data in the database the script to be run is: 
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
- ML pipeline: to load data from DB, train classifier and save the classifier as a pickle file the script to be run is: 
  python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
- The classification web app and the data visualizations are available in the Flask web app, which can be accessed at the address http://0.0.0.0:3001/ after running the script     python run.py in the app folder.
  The web app consists of a search bar where messages can be inputted for classification and descriptive visualisations of the dataset used for this project.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit for the data and the project initiative goes to Figure Eigth and Udacity.
