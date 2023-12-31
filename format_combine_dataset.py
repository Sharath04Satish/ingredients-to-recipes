# -*- coding: utf-8 -*-
"""Format_Combine_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1joBeqxBwu3vTwqkrJYZAJc-VKgtlqzGl
"""
import pandas as pd
import random
import numpy as np

department_dataset = pd.read_csv('/content/drive/MyDrive/Data and models/CS6350 - Many-to-Many/departments_dataset.csv')
department_dataset = department_dataset.drop(labels=[1,7,8,9,10,12,16,17,20], axis = 0)
print("\nDepartment Dataset:\n",department_dataset)

products_dataset = pd.read_csv('/content/drive/MyDrive/Data and models/CS6350 - Many-to-Many/products_dataset.csv')
product_dataset = products_dataset.drop(columns='aisle_id', axis=1)
print("\n\nProducts Dataset: \n", products_dataset.head)

orders_dataset = pd.read_csv('/content/drive/MyDrive/Data and models/CS6350 - Many-to-Many/orders_dataset.csv')
print("\n\nOrders Dataset:\n", orders_dataset.head )

products_department_dataset = combined_df = pd.merge(products_dataset, department_dataset, on='department_id')
all_product_ids = set(products_department_dataset['product_id'])
filtered_orders_dataset = orders_dataset[orders_dataset['product_id'].isin(all_product_ids)]

merged_dataset = orders_dataset.merge(products_department_dataset, on='product_id')
order_product_dataset = merged_dataset[['order_id', 'product_id', 'product_name']]

grouped_data = order_product_dataset.groupby('order_id').agg({
    'product_id': list,
    'product_name': list
}).reset_index()
grouped_data.rename(columns={'product_id': 'product_ids', 'product_name': 'product_names'}, inplace=True)
grouped_data.drop(columns= 'product_ids', inplace = True)
grouped_data = grouped_data[grouped_data['product_names'].apply(lambda x: 5 <= len(x) <= 15)]

final_data = grouped_data.sample(n=10000)

final_data['user_id'] = np.nan

while final_data['user_id'].isnull().any():
    num_orders = random.randint(1, 100)
    orders_sample = final_data.sample(n=num_orders)
    user_id = random.randint(1, 1000)

    update_condition = final_data['user_id'].isnull() & final_data['order_id'].isin(orders_sample['order_id'])
    final_data.loc[update_condition, 'user_id'] = user_id

print(final_data.head)

max_products_order = final_data.loc[final_data['product_names'].apply(len).idxmax()]

max_products_length = len(max_products_order['product_names'])

print("Order ID:", max_products_order['order_id'])
print("User IDs:", max_products_order['user_id'])
print("Product Names:", max_products_order['product_names'])
print("Number of Products:", max_products_length)

final_data.to_csv('/content/drive/MyDrive/Data and models/CS6350 - Many-to-Many/Final_orders_dataset.csv',index = True)