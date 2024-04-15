# Recipe-Data-Analysis

[The Recipe dataset](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m) from Kaggle contains over 2 million recipes from all over the world. The dataset contains the dish name, ingredients used and cooking instructions denoted in "directions" column. The author also provides a NER column that applies Named Entity Recognition (NER) on the ingredients and gives the base name of the ingredients used without the brands and quantity used.

This project aims to explore the possibilities opened by the below mentioned questions:

• Question 1: Can clustering techniques be employed to categorize recipes based on
their ingredient profiles and identify underlying patterns?

• Question 2: What are the very common ingredients used together?

• Question 3: Given the clusters, and a list of ingredients the user has and their
preferred dishes, can dishes be recommended to the user based on the user’s taste?

Since the decompressed dataset is over 2GB, Apache Spark MLlib will be used to analyse
the dataset.

The webapp for Question 3 can be accessed here - https://rit-ctrl-recipe-data-analysis-project.streamlit.app/

## Scala and Spark requirements:

- Java: 1.8.0
- Scala: 2.11.8
- Hadoop: 3.0.3
- Spark: 2.3.1
- Maven: 3.3.9

## Python requirements:
- streamlit==1.33.0
- pandas==2.1.0

## How to run

Q1a - 

```console
spark-submit --class project.embedding target/assignments-1.0.jar --input recipes_data.csv --output project-q1-vectors --vectorsize 10
```
 
Q1b  - 

```console
spark-submit --class project.kmeans target/assignments-1.0.jar --input project-q1-vectors/part-00000 --output project-q1-kmeans --k 2  
``` 

Q2 

```console
spark-submit --class --driver-memory 2g
project.itemset_mining target/assignments-1.0.jar --input recipes_data.csv --output project-q2-itemset --min-support 0.01 --min-confidence 0.01
```

Q3:
```console
streamlit run project_streamlit.py

```
## Project structure

[project](project) - the source code for Spark scripts written in Scala

[project.ipynb](project.ipynb) - Jupyter notebook used for reading the outputs from the Spark scripts and saving them in a useful format

[project_streamlit.py](project_streamlit.py) - script for deploying webapp for Question 3


