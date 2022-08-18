import os
from flask import Flask, render_template,url_for
from flask import request
from query import query_processing
import pandas as pd
import time
import pickle

app = Flask(__name__)
app.config["DEBUG"]= True


@app.route("/")
def index():
    '''
    Renders the starting query page.
    '''
    return render_template("index.html",docs=[]);   


@app.route('/search' , methods=['POST'])
def search():
    ''' 
    Function to be called after query is inserted to return the results.
    Data structure used includes dictionary and lists.
    Renders the final ranked documents.
    '''
    print("Time required for querying")
    start_time = time.time()
    ranks= query_processing(request.form['query'])
    file = open("book_description.obj",'rb')
    df = pickle.load(file)
    file.close()
    docs=[]
    for i in range(0,10) :
        docs+=[df.iloc[ranks[i][1]].to_dict()]
    docs = [[values for keys, values in docs[x].items() ]for x in range(0,10) ]
    print(docs)
    print("--- %s seconds ---" % (time.time() - start_time))
    return render_template("index.html",docs=docs);  
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)

#  ,docs=[["Movie 1","Some year", "something else","another attribute"]]