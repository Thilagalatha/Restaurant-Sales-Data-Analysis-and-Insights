# project.py
# Restaurant Sales Data Analysis and Insights
# Predict Order Total using Linear Regression & Deploy with Gradio


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import gradio as gr

# STEP 2: Load Dataset
df = pd.read_csv('restaurant_sales_data.csv')

# STEP 3: EDA & Basic Info
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.info()
print(df.describe())

# STEP 4: Data Cleaning
# Fill missing values
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop irrelevant columns
df.drop(['Order ID','Customer ID','Order Date'], axis=1, inplace=True)

# STEP 5: Visualizations (EDA)
plt.figure(figsize=(8,5))
sns.histplot(df['Order Total'], kde=True)
plt.title('Distribution of Order Total')
plt.xlabel('Order Total')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Category', y='Order Total', data=df)
plt.title('Category vs Order Total')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Payment Method', y='Order Total', data=df)
plt.title('Payment Method vs Order Total')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(6,5))
sns.scatterplot(x='Price', y='Order Total', data=df)
plt.title('Price vs Order Total')
plt.xlabel('Item Price')
plt.ylabel('Order Total')
plt.show()

plt.figure(figsize=(6,5))
sns.scatterplot(x='Quantity', y='Order Total', data=df)
plt.title('Quantity vs Order Total')
plt.xlabel('Quantity')
plt.ylabel('Order Total')
plt.show()

# STEP 6: Target & Features + Encoding
target = 'Order Total'
features = df.columns.drop(target)

categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 7: Train-Test Split & Model
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# STEP 8: Predict New Input
new_order = {
    'Category': 'Main Course',
    'Item': 'Chicken Biryani',
    'Price': 250.0,
    'Quantity': 2,
    'Payment Method': 'Credit Card'
}

new_df = pd.DataFrame([new_order])
df_temp = pd.concat([df.drop(target, axis=1), new_df], ignore_index=True)
df_temp_encoded = pd.get_dummies(df_temp, drop_first=True)
df_temp_encoded = df_temp_encoded.reindex(columns=X.columns, fill_value=0)
new_input_scaled = scaler.transform(df_temp_encoded.tail(1))
predicted_sales = model.predict(new_input_scaled)
print("Predicted Order Total:", round(predicted_sales[0],2))

# STEP 9: Deployment with Gradio
def predict_order_total(Category, Item, Price, Quantity, Payment_Method):
    input_data = {
        'Category': Category,
        'Item': Item,
        'Price': float(Price),
        'Quantity': float(Quantity),
        'Payment Method': Payment_Method
    }
    input_df = pd.DataFrame([input_data])
    df_temp = pd.concat([df.drop(target, axis=1), input_df], ignore_index=True)
    df_temp_encoded = pd.get_dummies(df_temp, drop_first=True)
    df_temp_encoded = df_temp_encoded.reindex(columns=X.columns, fill_value=0)
    scaled_input = scaler.transform(df_temp_encoded.tail(1))
    prediction = model.predict(scaled_input)
    return round(prediction[0],2)

inputs = [
    gr.Dropdown(df['Category'].unique().tolist(), label="Food Category"),
    gr.Dropdown(df['Item'].unique().tolist(), label="Menu Item"),
    gr.Number(label="Item Price"),
    gr.Number(label="Quantity Ordered"),
    gr.Dropdown(df['Payment Method'].unique().tolist(), label="Payment Method")
]

output = gr.Number(label="Predicted Order Total")

gr.Interface(
    fn=predict_order_total,
    inputs=inputs,
    outputs=output,
    title="Restaurant Sales Predictor",
    description="Enter order details to predict the total bill amount."
).launch()
