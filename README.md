# Recipe-Data-Analysis

Java: 1.8.0
Scala: 2.11.8
Hadoop: 3.0.3
Spark: 2.3.1
Maven: 3.3.9

Python requirements
streamlit==1.33.0
pandas==2.1.0


Run commands

Q1a - spark-submit --class project.embedding target/assignments-1.0.jar --input recipes_data.csv --output project-q1-vectors --vectorsize 10
 
Q1b  - spark-submit --class project.kmeans target/assignments-1.0.jar --input project-q1-vectors/part-00000 --output project-q1-kmeans --k 2   

Q2 spark-submit --class --driver-memory 2g
project.itemset_mining target/assignments-1.0.jar --input recipes_data.csv --output project-q2-itemset --min-support 0.01 --min-confidence 0.01