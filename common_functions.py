from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from spacy import load
from functools import reduce
import nltk
import dill
from constants import num_recommendations
import numpy as np
from random import randint
from constants import remove_words

# nltk.download('wordnet')
# nltk.download('punkt')

spacy_model = load(r"./output/model-best")
lemmatizer = WordNetLemmatizer()


def read_csv_data(filename):
    data = pd.read_csv("data/{0}".format(filename))
    return data


def lemmatize_words(word):
    if len(lemmatizer.lemmatize(word)) > 2:
        return lemmatizer.lemmatize(word)
    else:
        return ""


def format_product_names_from_data(df, index=None):
    if index is not None:
        random_row = df.loc[index, "product_names"]
    else:
        random_row = df.loc[randint(0, df.shape[0] - 1), "product_names"]
    random_row = random_row.replace("[", "").replace("]", "").replace("'", "")
    random_row = random_row.split(",")

    return [val.strip() for val in random_row]


def format_ingredients(ingredients):
    formatted_ingredients = set()
    for ingredient in ingredients:
        unique_values = list()
        doc = spacy_model(ingredient)

        for value in doc.ents:
            if str(value) != " " and str(value).lower() not in unique_values:
                unique_values.append(str(value).lower())

        name = " ".join(unique_values)
        name = " ".join(name.split())
        name = name.replace(",", "")
        if name:
            stemmed_item = reduce(
                lambda x, y: x + " " + y,
                map(lemmatize_words, nltk.word_tokenize(name)),
            )

            formatted_ingredients.add(stemmed_item)

    return formatted_ingredients


def getTfIdfModel():
    parsed_ingredients = list()
    with open("data/simplified_ingredient_dataset.json") as data_file:
        data = json.load(data_file)
        for v in tqdm(list(data.values())):
            parsed_ingredients.append(", ".join(v["Ingredients"]))

    column_data = pd.Series(parsed_ingredients)
    df = pd.DataFrame({"ingredients": column_data})

    tfIdfModel = TfidfVectorizer()
    tfIdfModel.fit(df["ingredients"])
    tfIdfEncodings = tfIdfModel.transform(df["ingredients"])

    with open("data/tfIdfModel.dill", "wb") as dill_file:
        dill.dump(tfIdfModel, dill_file)

    with open("data/tfIdfEncodings.dill", "wb") as dill_file:
        dill.dump(tfIdfEncodings, dill_file)


def RecSys(items_in_order):
    with open("data/tfIdfModel.dill", "rb") as dill_file:
        tfIdfModel = dill.load(dill_file)

    with open("data/tfIdfEncodings.dill", "rb") as dill_file:
        tfIdfEncodings = dill.load(dill_file)

    # Create a function from rrd file to format the input data from the file.
    formatted_ingredients = format_ingredients(items_in_order)
    ingredients_parsed = " ".join(formatted_ingredients)
    ingredients_tfidf = tfIdfModel.transform([ingredients_parsed])

    # calculate cosine similarity between actual recipe ingreds and test ingreds
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfIdfEncodings)
    scores = list(cos_sim)

    # Filter top N recommendations
    top_n_recommendations = get_recommendations(scores)
    return top_n_recommendations, scores


def get_recommendations(scores):
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :num_recommendations
    ]


def format_recommendations_df(top_n_recommendations, scores, items_in_order):
    recommendation_table = pd.DataFrame(
        columns=["recipe_name", "ingredients", "instructions", "picture", "score"]
    )
    recipe_count = 0
    ingredients = list()

    with open("./data/simplified_ingredient_dataset.json") as data_file:
        data = json.load(data_file)

    for nth in top_n_recommendations:
        recommendation_table.at[recipe_count, "recipe_name"] = list(data.keys())[
            nth
        ].strip()
        ingredients.append(list(data.items())[nth][1]["Ingredient_Descriptions"])
        recommendation_table.at[recipe_count, "ingredients"] = list(data.items())[nth][
            1
        ]["Ingredient_Descriptions"]
        recommendation_table.at[recipe_count, "instructions"] = list(data.items())[nth][
            1
        ]["Instructions"]
        recommendation_table.at[recipe_count, "picture"] = list(data.items())[nth][1][
            "Picture"
        ]
        recommendation_table.at[recipe_count, "score"] = scores[nth][0]

        recipe_count += 1

    return (recommendation_table, ingredients)


def get_orders_by_user(df):
    random_user_id = df.loc[randint(0, df.shape[0] - 1), "user_id"]
    return df[df["user_id"] == random_user_id]


def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_base_items(items):
    base_ingredients = set()

    for item in items:
        item = item.split(" ")

        for value in item:
            if is_numeric(value) == False and value not in remove_words:
                base_ingredients.add(lemmatize_words(value.lower().replace(",", "")))

    return base_ingredients
