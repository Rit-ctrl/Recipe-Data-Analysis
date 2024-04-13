import streamlit as st
# from streamlit_option_menu import option_menu
import pandas as pd
import dask.dataframe as dd
import ast


st.title("Recipe Recommendation System")
@st.cache_data
def load_data():
    '''
    Returns the transformed df, list of dishes, and master list of ingredients
    '''
    # return pd.read_csv("project-data/dishes.csv")
    with open('project-data/master-ingredients.txt','r') as file:
        lines = file.readlines()
    master_ingredients = [line.strip() for line in lines]
    df = pd.read_csv("project-data/transformed_sample.csv")
    return df,dd.read_csv("project-data/dishes.csv"),master_ingredients

with st.spinner("Loading data..."):
    df,dishes,master_ingredients = load_data()
    # master_ingredients

# @st.cache_data
# def get_options(df):
#     return df['title'].unique().tolist()

options = dishes["title"].unique().compute().tolist()

# print(options[:5])
# options = get_options()


selected_ingredients = st.multiselect("Choose Ingredients available", master_ingredients)

selected_options = st.multiselect("Choose your favourite dishes", options)

submit_button = st.button("Submit")


# Display the selected options
if submit_button and selected_ingredients and selected_options:
    st.write("Selected Options:")
    matched_cluster = df[df['title'].apply(lambda x:x in selected_options)]['cluster'].mode().values[0]

    # st.write(matched_cluster)

    matched_dishes = df[(df['cluster'] == matched_cluster)]
    # matched_dishes = matched_dishes[matched_dishes['NER'].apply(lambda x:len(set(ast.literal_eval(x)).intersection(selected_ingredients)) > 0 )]
    matched_dishes['intersection_score'] = matched_dishes['NER'].apply(lambda x:len(set(ast.literal_eval(x)).intersection(selected_ingredients)))
    matched_dishes['intersection_score'] = matched_dishes['intersection_score'] / matched_dishes['NER'].apply(lambda x:len(set(ast.literal_eval(x)).union(selected_ingredients)))
    st.write(matched_dishes[['title','ingredients','directions','link','NER','site','intersection_score']].sort_values(by = 'intersection_score',ascending = False).head())
    # for option in selected_options:
    #     st.write(f"- {option}")
elif not selected_ingredients:
    st.write("Ingredients cannot be empty")
elif not selected_options:
    st.write("Liked dishes cannot be empty")