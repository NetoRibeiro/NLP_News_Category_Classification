"""
    DATA8001 - Assignment 2 2021
    
    description: All Assignment 2 functions required to reproduce results
    
    version   1.0       ::  2021-03-22  ::  started version control

                                            ----------------------------------
                                                GENERIC FUNCTIONS
                                            ----------------------------------
                                            
                                            + data_etl
                                            + load_run_model
                                            
                                            ----------------------------------
                                                USER DEFINED FUNCTIONS
                                            ----------------------------------
                                            + 
"""

##########################################################################################
##########################################################################################
#
#   IMPORTS
#
##########################################################################################
##########################################################################################


import re
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime as dt


##########################################################################################
##########################################################################################
#
#   GENERIC FUNCTIONS
#
##########################################################################################
##########################################################################################

def data_etl(student_id):
    """
        Load original data from news files and clean
        
        :param str student_id:
            The student_id to unzip the news files.

        :return: Processed (cleaned) pandas dataframe.
        :rtype: pandas.core.frame.DataFrame
    """
    try:

        print(f'cleaning data for {student_id} ...')
        
        df_processed = None
        
        #
        # 1. TODO - load the news files into a dataframe with the columns: ['news_headline', 'news_article']
        #
        dir_cwd = os.path.abspath(os.path.pardir)
        article_files = get_etl_stage_a(student_id, dir_cwd)
        print(f'Stage A')
        #
        # 2. TODO - clean the data
        #
        df_processed = get_etl_stage_b(article_files)
        print(f'Stage B')
        #
        # 3. return processed (clean) data with columns: ['news_headline', 'news_article']
        #
        
        return (df_processed)
        
    except Exception as ex:
        raise Exception(f'data_etl :: {ex}')


def load_run_model(model_id, student_id, news_headline, news_article):
    """
        Load the specified student pickle model and use the ML model to predict a new category based on the news headline and article provided.

        :param str model_id:
            The model_id to load the model file path of the pickled model.
            ['model_1', 'model_2', 'model_3']
        :param str student_id:
            The student_id to load the model file path of the pickled model.
        :param str news_headline:
            The news headline
        :param str news_article:
            The news article
            
        :return: Model object and model predicted category (i.e., news category)
        :rtype: object, str
    """
    try:
        if model_id not in ['model_1', 'model_2', 'model_3', 'model_4']:
            raise Exception('invalid model id')
            
        print(f'loading and running the {model_id} model for {student_id} ...')

        model = None
        news_category = ''
        

        #
        # 1. load the correct pickled model
        #
        # model_name = f'{student_id}/model/{student_id}_{model_id}.pkl'
        model_name = f'D:/Neto/CIT/Data Science and Analytics/Project_002/R00206995/model/R00206995_{model_id}.pkl'
        #
        # 2. Pre-Process the news_headline and news_article values if required
        #
        model, news_category = unseen_transformation(model_name, news_headline, news_article)
                
        #
        # 3. run the model to predict the new category
        #
        
        return (model, news_category)
        
    except Exception as ex:
        raise Exception(f'load_run_model :: {ex}')
        
        
class Data8001():
    """
        Data8001 model & transformation class
    """               
        
    #############################################
    #
    # Class Constructor
    #
    #############################################
    
    def __init__(self, transformations:list, model:object):
        """
            Initialse the objects.

            :param list transformations:
                A list of any data pre-processing transformations required
            :param object model:
                Model object
        """
        try:
            
            # validate inputs
            if transformations is None:
                transformations = []
            if type(transformations) != list:
                raise Exception('invalid transformations object type')
            if model is None:
                raise Exception('invalid model, cannot be none')                                
            
            # set the class variables
            self._transformations = transformations
            self._model = model
            
            print('data8001 initialised ...')
            
        except Exception as ex:
            raise Exception(f'Data8001 Constructor :: {ex}')
            
    #############################################
    #
    # Class Properties
    #
    #############################################
    
    def get_transformations(self) -> list:
        """
            Get the model transformations
            
            :return: A list of model transformations
            :rtype: list
        """
        return self._transformations
    
    def get_model(self) -> object:
        """
            Get the model object
            
            :return: A model object
            :rtype: object
        """
        return self._model

        
##########################################################################################
##########################################################################################
#
#   USER DEFINED FUNCTIONS
#
#   provide your custom functions below this box.
#
##########################################################################################
##########################################################################################

def get_etl_stage_a(student_id, dir_cwd):
    """
    Purpose of this function:
        Convert all articles files into a file
    :return: List of loaded files
    :rtype: list
    """

    # Access file directory
    original_dir_cwd = os.getcwd()
    files_dir_cwd = f'{dir_cwd}\\{student_id}\\data\\files'
    # files_dir_cwd = f'data/files'
    files = os.listdir(files_dir_cwd)
    os.chdir(files_dir_cwd)
    
    # Initialize an empty list and column names to not add on the list
    columns_original = ["", "REPORTER", "DATE", "CATEGORY", "HEADLINE", "ARTICLE"]
    article_files = []

    # Read each file into a list
    for file_without_ext in files:
        # No load the readme file
        find_readme_file = file_without_ext.replace(dir_cwd, "")
        if find_readme_file != "_README":
            each_file = open(file_without_ext, 'r', encoding='utf-8')
            file_read = each_file.read().replace("</", "|").replace(">", "|").replace("<", "|").replace("\n", "").split('|')
            list_file = []
            for phrase in file_read:
                if phrase not in(columns_original):
                    list_file.append(phrase)
            article_files.append(list_file)
            each_file.close()

    os.chdir(original_dir_cwd)
    return article_files


def get_etl_stage_b(article_files):
    """
    Purpose of this function:
        Convert articles file into a data frame
    :param str article_files:
            Receive the RAW loaded files list.

    :return: Processed (cleaned) pandas dataframe.
    :rtype: pandas.core.frame.DataFrame
    """

    # Create a data frame and add the list into
    columns = ["reporter", "date", "news_category", "news_headline", "news_article"]
    df = pd.DataFrame(article_files, columns=columns)
    # Create Processed file to a drive
    # df.to_csv(f'data/R00206995_processed.csv', index=False)
    df_processed = df[["news_headline", "news_article"]].copy()
    return df_processed


def nouns_adj(text_in):
    """
    Purpose of this function:
        Given a string of text, tokenize the text and pull out only the nouns and adjectives.
    :param string text_in:
        String corpus
    :return: tokenized corpus with only  NOUNS and ADJECTIVES
    """
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    try:
            
        tokens = [re.sub(r'\d+', ' ', word).strip() for word in text_in.split()]
        tokens = [word for word in tokens if word != '']
        text_in = (' '.join(tokens))

        tokens = [re.sub(r'\W', ' ', word).strip() for word in text_in.split()]
        tokens = [word for word in tokens if word != '']
        text_in = (' '.join(tokens))
        
        is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
        tokenized = word_tokenize(text_in)
        nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
        phrase_NN_JJ_return = ' '.join(nouns_adj)

    except Exception as ex:
        phrase_NN_JJ_return = text_in
        print(f'Excption occur when try to get POS TAG:\terror::{ex}')

    return phrase_NN_JJ_return


def unseen_transformation(model, news_headline, news_article):
    """
    Purpose of this function:
        Transform the String
    :param model:
        Pickle model to load
    :param string news_headline:
        column content
    :param string news_article:
        column content
    :return: Model and Predicted Y (News Category)
    """

    # for processing
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    from nltk.corpus import stopwords
    try:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
    except Exception as ex:
        print(f'Error when try to download NLTK Stopwords:\terror::{ex}')

    # Combine Features into a string corpus
    try:
        text_in = news_headline + " " + news_article
    except Exception as ex:
        text_in = news_article
        print(f'Exception when try to combine strings:\terror::{ex}')
    
    # Apply Data transformation:
    try:
        # Create a list from the string
        list_text_in = text_in.split()

        # Removing Stopwords
        list_stopwords = stopwords.words("english")
        list_text_in = [word_in for word_in in list_text_in if word_in not in list_stopwords]

        # Applying Lemmatization
        w_lem = WordNetLemmatizer()
        list_text_in = [w_lem.lemmatize(word_in) for word_in in list_text_in]

        # Return a String
        list_text_in = " ".join(list_text_in)
    except Exception as ex:
        list_text_in = [text_in]
        print(f'Exception when try to apply data transformation:\terror::{ex}')

    try:
        corpus = nouns_adj(list_text_in)
    except:
        corpus = list_text_in
    
    try:
        model = pickle.load(open(model, 'rb'))
    except Exception as msg_ex:
        print(f'Sorry, something went wrong when try to load Pickle model:\t{model}\terror:: {msg_ex}')
    
    X_predict = [corpus]

    try:
        predicted = model.predict(X_predict)
    except Exception as msg_ex:
        print(f'Sorry, something went wrong when try to Predict:\t{X_predict}\terror:: {msg_ex}')
        
    return model, predicted

