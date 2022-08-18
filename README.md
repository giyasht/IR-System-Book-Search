# IR-System_Book-Search
### About
This is a repository for the development of an IR system to search for books from the database provided here.


## DATABASE.CSV
This file contains the CSV description of the books we are using for the search system.

## PREPROCESS.PY
This file creates .obj file after processing the csv data from database.csv file after tokenizing, stemming, lemmatizing it.
It also creates the posting lists for the titles and the inverted index as well.

## TF-IDF.PY
This file contains the code for the term-frequency and invers-document-frequency of the inverted indexed file to rank the files.

## QUERY.PY
This file contains the code for the query processing.
Also it is responsible for presenting the top 10 results of the search.

## SERVER.PY
This file contains the code for the deployment and start of the IR system on web.

## INDEX.HTML / MAIN.CSS
Contains the front end code for the site.
