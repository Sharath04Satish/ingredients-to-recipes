import json
from tqdm import tqdm
from common_functions import format_ingredients

json_data = dict()


with open("./data/recipes_dataset.json") as data_file:
    data = json.load(data_file)
    for index, v in enumerate(tqdm(list(data.values()))):
        formatted_ingredients = format_ingredients(v["ingredients"])

        recipe_key = v["title"].strip()
        json_data[recipe_key] = {
            "Instructions": v["instructions"],
            "ingredients": list(formatted_ingredients),
            "Picture": v["picture_link"],
            "Ingredient_Descriptions": v["ingredients"],
        }


with open("./data/simplified_ingredient_dataset.json", "w") as write_file:
    json.dump(json_data, write_file)
