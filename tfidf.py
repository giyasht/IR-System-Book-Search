import time
import nltk
import pandas as pd
import numpy as np
import pickle

from preprocess import COUNT

def preprocess_tfidf(inverted_index, processed_data, length):
    ''' 
    Function to create tf-idf for all the documents. 
    Data structures used includes Dictionary and Dataframes. 
    Loads the Indexing list and Term frequencies and calculates the BM25 tf-idf score for all the terms in a given document.
    Final tf-idf scores are getting stored in a pickle file.
    BM 25 tf-idf =  idf* tf*((k+1) )/(k*(1- b + b*((length of document)/(avg length of document))) * 100
    where k and b are fixed parameters and tf, idf are term frequency and inverse document frequency .
    tf = no of times term occuring a document/total length of document
    idf = log(no of documents/ no of documents containing that term)
    Perfomes the same procedure for the titles of the documents, calculating the BM25 tf-idf score for each term in the title as well. 
    Returns tf-idf dictionary
    '''
    print("Time required to create tf-idf for corpus")
    start_time = time.time()
    no_of_doc = COUNT

    # Function to load a pickle file
    file = open(processed_data, 'rb')
    df = pickle.load(file)
    file.close()

    # Loading term frequencies
    d_df = df.to_dict()
    d_df = d_df[length]

    # Average length of all documents
    avg_length = df[length].mean()

    # Loading indexing list
    file = open(inverted_index, 'rb')
    ii_df = pickle.load(file)
    file.close()

    ii_df = ii_df.to_dict()
    ii_df = ii_df['PostingList']

    k = 1.75  # Parameter for BM25
    b = 0.75  # Parameter for BM25

    tf_idf_dict = {}

    # Calculating tf-idf
    for doc in range(0, no_of_doc):
        doc_dict = {}
        for key, value in df['Frequency'][doc].items():
            if key == 'nan' or key == 'null':
                continue
            tf = (value/d_df[doc])
            idf = np.log(no_of_doc/(ii_df[key]))
            doc_dict[key] = idf*(tf*(k+1))/(tf + k *
                                            (1 - b + b*(df[length][doc]/avg_length))) * 100
        tf_idf_dict[doc] = doc_dict

    print("--- %s seconds ---" % (time.time() - start_time))
    # print(tf_idf_dict)
    return tf_idf_dict


# main function
tf_idf_dict = preprocess_tfidf(
    "inverted_index.obj", "processed_data.obj",  "Length")
file_pointer = open("tf-idf.obj", "wb")
pickle.dump(tf_idf_dict, file_pointer)
file_pointer.close()

tf_idf_title_dict = preprocess_tfidf(
    "inverted_index_title.obj", "processed_data_title.obj",  "TitleLength")
file_pointer = open("tf-idf_title.obj", "wb")
pickle.dump(tf_idf_title_dict, file_pointer)
file_pointer.close()