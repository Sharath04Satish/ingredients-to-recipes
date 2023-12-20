# many-to-many

A machine learning algorithm that suggests food recipes based on ingredients in customers' carts.

## Datasets

**departments_dataset.csv**
<br>
Columns - department_id, department

**orders_dataset.csv**
<br>
Columns - order_id, product_id, add_to_cart_order, reordered

**products_dataset.csv**
<br>
Columns - product_id, product_name, aisle_id, department_id

**recipes_dataset.json**
<br>
Columns - ingredients, picture_link, instructions, title

## Data Pre-Processing Procedure

1. Filter appropriate departments from departments.csv which align with preparing dishes with food ingredients.
2. Filter products from products.csv, based on the departments filtered from the first step.
3. Filter products from products.csv if products were never ordered in order_products_prior.csv.
4. Prepare a new dataset by mapping order_id, product_id, and product_name.
5. Determine a way to extract ingredients from lists of ingredients for the recipes.
6. Map dishes along with its ingredients and its cooking instructions.
7. Filter the dataset from step 4 based on the ingredient list from step 5.

## What we're trying to achieve

- After pre-processing, we'll have two datasets, one for order transactions for products and another dataset which contains recipes.
- We’ll be developing a recommendation system using a collaborative filtering algorithm which recommends dishes based on products present in customers’ shopping carts.
- Future work includes recommending remaining ingredients to complete a dish selected by customers.

## Datasets and Model Outputs for NER

Since the datasets and model outputs are huge (of the size greater than 2.5GB), they are stored in the Google Drive at the following link, https://drive.google.com/drive/folders/1uXVtDFUxLQxarNXNXLFloE5wnnUKMB8F?usp=sharing

Previously we were using a different repository, where we tried pushing files over 100MB, and starting getting errors while pushing the code, and hence we had to create a new repository and move the code here.

## Execution Procedure

- If you wish to run the code, in order to train the custom Named Entity Recognition model, use the following command. Replace python with python3 if using python throws an error. This will create a folder named "output" at the root folder.
  "python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./train.spacy"
- Execute option 4 of get_recommendations.py file using "python get_recommendations.py" to generate the tf - idf model. Then select any option to display the output.
