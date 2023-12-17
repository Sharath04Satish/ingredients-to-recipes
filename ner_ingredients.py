import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
import re
from constants import ner_training_dataset, remove_words, exluded_characters

# Enable GPU acceleration using CUDA cores.
spacy.require_gpu()

# Read training data, drop non-existent rows from input and name columns.
training_data = pd.read_csv(ner_training_dataset)
training_data["input"] = training_data["input"].astype(str)
training_data["name"] = training_data["name"].astype(str)
training_data = training_data.dropna(axis=0, subset=["input", "name"])


def format_training_data(df, col):
    # Format training data to remove unsupported characters and stop words related to recipes.
    for index, row in df.iterrows():
        formatted_row = str(row[col])
        for character in exluded_characters:
            formatted_row = formatted_row.replace(character, "")

        curr_row = formatted_row.split()
        if len(curr_row) > 1:
            resultwords = [
                word for word in curr_row if word.lower() not in remove_words
            ]
            formatted_row = " ".join(resultwords)
        else:
            formatted_row = " "
        if formatted_row == " ":
            df.at[index, col] = None
        else:
            df.at[index, col] = formatted_row
    return df


# Drop non-existent rows from input and name columns after formatting the rows.
formatted_training_data = format_training_data(training_data, "name")
formatted_training_data = formatted_training_data.dropna(
    axis=0, subset=["input", "name"]
)


def generateEntity(ingredient_description, ingredient_list, named_entity):
    unique_ingredients = {"entities": []}
    if len(ingredient_list) == 1:
        regex = re.compile(ingredient_list[0])
        entity_match = regex.search(ingredient_description)
        if entity_match is not None:
            unique_ingredients["entities"] = [
                (entity_match.start(), entity_match.end(), named_entity)
            ]

        return unique_ingredients["entities"]
    else:
        for index in range(len(ingredient_list)):
            regex = re.compile(ingredient_list[index])
            entity_match = regex.search(ingredient_description)
            if entity_match is not None:
                if index == 0:
                    unique_ingredients["entities"] = [
                        (entity_match.start(), entity_match.end(), named_entity)
                    ]
                else:
                    unique_ingredients["entities"].append(
                        (entity_match.start(), entity_match.end(), named_entity)
                    )

    return unique_ingredients["entities"]


def generateTrainingData(
    dataframe, ingredient_description, ingredient_name, named_entity
):
    spacy_training_data = list()
    subset = dataframe[[ingredient_description, ingredient_name]]

    for index in range(len(dataframe)):
        description, name = subset.iloc[index, 0], subset.iloc[index, 1]
        list_ingredients = name.split()

        unique_entities = {}
        is_invalid_row = False

        for ingredient in list_ingredients:
            if description == "nan" or ingredient == "nan":
                is_invalid_row = True
                continue
            if ingredient not in description:
                is_invalid_row = True
                continue

        if is_invalid_row == False:
            unique_entities["entities"] = generateEntity(
                description, list_ingredients, named_entity
            )
            spacy_training_data.append((description, unique_entities))

    return spacy_training_data


spacy_training_data = generateTrainingData(
    formatted_training_data[:50000], "input", "name", "INGREDIENT"
)

db = DocBin()
base_nlp_model = spacy.blank("en")

for description, annotation in tqdm(spacy_training_data):
    doc = base_nlp_model.make_doc(description)
    entities = []
    for start, end, label in annotation["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is not None:
            entities.append(span)

    # Handle
    filtered = spacy.util.filter_spans(entities)
    doc.ents = filtered
    db.add(doc)

db.to_disk("./train.spacy")
