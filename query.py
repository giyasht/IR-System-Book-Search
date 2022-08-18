import time
import nltk
import pandas as pd
import numpy as np
import pickle

from preprocess import COUNT
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from collections import defaultdict
import heapq


def load(doc):
    ''' 
    Function to load a pickle file 
    '''
    file = open(doc,'rb')
    df = pickle.load(file)
    file.close()
    return df

def query_processing(query):
    ''' 
    Function to calculate ranking of documents.
    Data structure used includes Lists, Heaps, Dictionary, and Dataframes.
    Preprocessing the query in a way similar to the documents by converting into lower case, tokenizing and stemming.
    Calculates the term frequency for query and calculating it's tf-idf.
    Calculates the cosine similarity between the documents and query and ranking the same using a heap.
    Takes query as parameter and returns a list of top 10 highest ranked documents.
    '''
    no_of_doc = COUNT +1 # Actual number of documents + 1

    # All alphabets to lower case
    query= query.lower() 

    # Preprocessing query removing unnecessary characters
    query = query.replace("\n"," ").replace("\r"," ")
    query = query.replace("'s"," ")
    punctuationList = '-./:;<=>?@[\\]^_{|}~!"#$%&\()*+,'
    x = str.maketrans(dict.fromkeys(punctuationList," "))
    query = query.translate(x)

    # Tokenize
    df=word_tokenize(query)
    query_length=len(df)
    df= [w for w in df if not w in stopwords.words('english') ]
    
    # Stemming
    ps = PorterStemmer() 
    df=[ps.stem(word) for word in df]
    
    # Term frequency 
    query_freq = defaultdict(lambda: 0)

    for token in df :
        query_freq[token] +=1

    # Calling helper functions to calculate tf-idf for query with documents and title as corpus
    q_ls, q_tf_idf = helper("inverted_index.obj", "processed_data.obj", "Length", no_of_doc, query_length, query_freq)
    q_ls_title, q_tf_idf_title = helper("inverted_index_title.obj", "processed_data_title.obj", "TitleLength", no_of_doc, query_length, query_freq)

    # Cosine similarity
    tf_idf = load("tf-idf.obj")
    tf_idf_title = load("tf-idf_title.obj")

    q_ls += q_ls_title
    q_ls = np.array(q_ls)
    norm_query= np.sqrt(np.sum(q_ls*q_ls))#mod for query

    rank_heap=[]
    for doc in range(0,no_of_doc-1) :
        val=0   # Dot-product
        epsilon=10e-9
        d_ls = [] # Store tf-idf values for a doc
        for term , value in tf_idf[doc].items() :
            if (term in q_tf_idf.keys()):
                val+=(value * q_tf_idf[term])
            d_ls+=[value]
        for term , value in tf_idf_title[doc].items() :
            if (term in q_tf_idf_title.keys()):
                val+=(value * q_tf_idf_title[term])
            d_ls+=[value]
        d_ls = np.array(d_ls)
        norm_doc = np.sqrt(np.sum(d_ls*d_ls)) #mod value for doc
        cosine_sim = val/(norm_doc*norm_query + epsilon)
        heapq.heappush(rank_heap, (cosine_sim, doc))
    req_doc= heapq.nlargest(10, rank_heap)

    return req_doc




def helper(inverted_index, processed_data, length, no_of_doc, query_length, query_freq):
    '''
    Function to calculate tf idf for query.
    Data structures used includes List, Dictionary and Dataframes.
    Calculates BM25 tf-idf for query in a way similar to document.
    Takes the corpus parameters and returns tf-idf values.
    '''
    # tf-idf for query
    ii_df = load(inverted_index)

    ii_df= ii_df.to_dict()
    ii_df=ii_df['PostingList']

    k=1.75 # parameter for BM25
    b=0.75 # parameter for BM25

    tdf = load(processed_data)
    

    avg_length= tdf[length].mean() #Average length of documents
    avg_length= ((no_of_doc-1)*avg_length + query_length)/no_of_doc

    q_tf_idf={} # Stores only the tf-idf values that are common with the bag of words
    q_ls=[] # Stores all the tf-idf values
    for key , value in query_freq.items() :
        tf = (value/query_length)
        if key in ii_df.keys(): 
            idf = np.log(no_of_doc/ii_df[key])
            q_tf_idf[key] = idf * ( tf * (k+1) ) / (tf + k * (1- b + b * (query_length / avg_length))) *100
        q_ls+= [np.log(no_of_doc) * ( tf * (k+1) ) / (tf + k * (1- b + b * (query_length / avg_length))) *100] # add the value of tf idf of each term to the list

    return q_ls, q_tf_idf

# print(query_processing("bar appear enjoy face inside group work"))         
