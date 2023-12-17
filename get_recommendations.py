from common_functions import (
    RecSys,
    getTfIdfModel,
    format_recommendations_df,
    read_csv_data,
    format_product_names_from_data,
    get_base_items,
)
import time
from tqdm import tqdm
import numpy as np
import random
from constants import num_recommendations

users_orders_df = read_csv_data("users_orders_dataset.csv")

while True:
    try:
        option = int(
            input(
                "What do you want to do today?\nSelect 1 to provide recipe recommendations to a new user\nSelect 2 to get accuracy scores over test dataset\nSelect 3 to get average computation times for top@5 recommendations\nSelect 4 to build the TF - IDF model\n"
            )
        )
        break
    except:
        print("Let's try that one more time.")

if option == 1:
    items_in_order = format_product_names_from_data(users_orders_df)

    top_n_recommendations, scores = RecSys(items_in_order)
    recommendations, ingredients = format_recommendations_df(
        top_n_recommendations, scores, items_in_order
    )
    print("The items in the cart are,\n{0}".format(items_in_order))
    print(
        "\nGiven the list of items in the cart, these are the recipes that you could work with, "
    )
    print(recommendations)

# elif option == 2:
#     df_by_user = get_orders_by_user(users_orders_df)
#     top_n_recommendations_user, scores_user = list(), list()

#     order_threshold = min(2, df_by_user.shape[0])

#     for index, row in tqdm(df_by_user[:order_threshold].iterrows()):
#         items_in_order = format_product_names_from_data(df_by_user, index)
#         top_n_recommendations, scores = RecSys(items_in_order)
#         top_n_recommendations_user.append(top_n_recommendations)
#         scores_user.append(scores)

#     print(top_n_recommendations_user)
#     print(scores_user)

elif option == 2:
    precision_at_k_scores = list()
    for _ in tqdm(range(500)):
        random_row_indice = random.randint(0, users_orders_df.shape[0] - 1)
        items_in_order = format_product_names_from_data(
            users_orders_df, random_row_indice
        )
        top_n_recommendations, scores = RecSys(items_in_order)
        recommendations, ingredients = format_recommendations_df(
            top_n_recommendations, scores, items_in_order
        )
        base_ingredients_input = get_base_items(items_in_order)
        relevance_count = 0

        for ingr in ingredients:
            base_ingredients_ingr = get_base_items(ingr)
            is_relevant = 0

            for product in base_ingredients_ingr:
                if product in base_ingredients_input:
                    is_relevant += 1

            if is_relevant > 2:
                relevance_count += 1

        precision_at_k_scores.append(relevance_count / num_recommendations)

    print(
        "The average precision@k score for {0} recommendations are, {1}".format(
            num_recommendations, np.mean(np.asarray(precision_at_k_scores))
        )
    )

elif option == 3:
    prediction_times = list()
    for _ in tqdm(range(500)):
        start_time = time.time()
        items_in_order = format_product_names_from_data(users_orders_df)

        top_n_recommendations, scores = RecSys(items_in_order)
        recommendations, ingredients = format_recommendations_df(
            top_n_recommendations, scores, items_in_order
        )
        end_time = time.time()
        prediction_times.append(end_time - start_time)

    print("The minimum prediction time is, {0}".format(min(prediction_times)))
    print("The maximum prediction time is, {0}".format(prediction_times))

elif option == 4:
    print("Running the TF IDF model to build the model and the encodings.")
    getTfIdfModel()
    print("The TF IDF model has been built successfully.")
