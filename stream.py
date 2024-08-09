from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from deep_translator import GoogleTranslator

app = FastAPI()

# Load the dataset
df = pd.read_csv("cleaned_data.csv")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Define a normalization function for text
def normalize_text_simple(text):
    if pd.isna(text):  # Handle NaNs
        return ''
    text = str(text)  # Convert to string
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Normalize and prepare the separate features
df['normalized_product_name'] = df['product_name'].apply(normalize_text_simple)
df['normalized_product_category'] = df['product_category_tree'].apply(normalize_text_simple)
df['normalized_brand'] = df['brand'].apply(normalize_text_simple)
df['normalized_description'] = df['description'].apply(normalize_text_simple)
df['normalized_extracted_specifications'] = df['extracted_specifications'].apply(normalize_text_simple)

# Initialize the TF-IDF Vectorizers with reduced features
tfidf_name = TfidfVectorizer(max_features=3000)
tfidf_category = TfidfVectorizer(max_features=3000)
tfidf_brand = TfidfVectorizer(max_features=3000)
tfidf_description = TfidfVectorizer(max_features=3000)
tfidf_specifications = TfidfVectorizer(max_features=3000)

# Fit and transform the features
tfidf_name_matrix = tfidf_name.fit_transform(df['normalized_product_name'])
tfidf_category_matrix = tfidf_category.fit_transform(df['normalized_product_category'])
tfidf_brand_matrix = tfidf_brand.fit_transform(df['normalized_brand'])
tfidf_description_matrix = tfidf_description.fit_transform(df['normalized_description'])
tfidf_specifications_matrix = tfidf_specifications.fit_transform(df['normalized_extracted_specifications'])

# Define weights for each feature
weights = {
    'name': 0.4,
    'category': 0.2,
    'brand': 0.15,
    'description': 0.15,
    'specifications': 0.1
}

def process_query(query):
    # Detect and translate the query if it's not in English
    detected_lang = detect(query)
    if detected_lang != 'en':
        query = GoogleTranslator(source=detected_lang, target='en').translate(query)

    # Normalize the query
    query = normalize_text_simple(query)
    return query

def get_weighted_tfidf_vector(query):
    query = normalize_text_simple(query)
    
    # Transform the query for each feature
    query_name_vector = tfidf_name.transform([query])
    query_category_vector = tfidf_category.transform([query])
    query_brand_vector = tfidf_brand.transform([query])
    query_description_vector = tfidf_description.transform([query])
    query_specifications_vector = tfidf_specifications.transform([query])
    
    # Compute weighted sum of feature vectors
    weighted_vector = (weights['name'] * query_name_vector +
                       weights['category'] * query_category_vector +
                       weights['brand'] * query_brand_vector +
                       weights['description'] * query_description_vector +
                       weights['specifications'] * query_specifications_vector)
    return weighted_vector

def calculate_discount_percentage(row):
    if row['retail_price'] > 0:
        return 100 * (1 - row['discounted_price'] / row['retail_price'])
    return float('nan')  # Return NaN if the retail_price is 0 or missing

def get_relevant_products(query, max_price, min_discount, min_rating, top_n=12):
    if not query.strip():  # Handle empty queries
        return []  # Return an empty list if query is empty
    query_vector = get_weighted_tfidf_vector(query)
    
    # Compute cosine similarity for each feature matrix
    similarities_name = cosine_similarity(query_vector, tfidf_name_matrix).flatten()
    similarities_category = cosine_similarity(query_vector, tfidf_category_matrix).flatten()
    similarities_brand = cosine_similarity(query_vector, tfidf_brand_matrix).flatten()
    similarities_description = cosine_similarity(query_vector, tfidf_description_matrix).flatten()
    similarities_specifications = cosine_similarity(query_vector, tfidf_specifications_matrix).flatten()
    
    # Combine similarities with weights
    combined_similarities = (weights['name'] * similarities_name +
                             weights['category'] * similarities_category +
                             weights['brand'] * similarities_brand +
                             weights['description'] * similarities_description +
                             weights['specifications'] * similarities_specifications)
    
    top_indices = combined_similarities.argsort()[-top_n:][::-1]  # Get indices of top_n products
    results = df.iloc[top_indices].copy()

    # Convert 'product_rating' to numeric, forcing errors to NaN
    results['product_rating'] = pd.to_numeric(results['product_rating'], errors='coerce')

    # Calculate discount percentages
    results['discount_percentage'] = results.apply(calculate_discount_percentage, axis=1)

    # Apply additional filters considering missing values
    filtered_results = results[
        ((results['discounted_price'].fillna(max_price + 1) <= max_price) | results['discounted_price'].isna()) &
        ((results['discount_percentage'].fillna(-1) >= min_discount) | results['discount_percentage'].isna()) &
        ((results['product_rating'].fillna(-1) >= min_rating) | results['product_rating'].isna())
    ]

    products = []
    for _, row in filtered_results.iterrows():
        # Ensure that 'image' column contains publicly accessible URLs
        try:
            image_urls = eval(row['image'])  # Convert the string to a list of URLs
        except:
            image_urls = []  # Handle cases where eval fails
        if image_urls:  # If the list is not empty
            products.append({
                "image": image_urls[0],  # Use the first image URL
                "name": row['product_name'],
                "price": row['discounted_price'],
                "url": row['product_url']
            })
    return products

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zepto Product Search System</title>
    <style>
        body {
            padding: 20px;
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f9;
            color: #333;
            display: flex;
            justify-content: center;
                        
            align-items: center;
            height: 100vh;
            margin: 20px;
            padding:20px;
        }
        .container {
            padding: 20px;
            max-width: 700px;
            width: 100%;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #3498db;
            margin-bottom: 20px;
        }
        p {
            color: #666;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }
        .form-group label {
            display: block;
            font-size: 16px;
            margin-bottom: 5px;
            color: #333;
        }
        input, select {
            padding: 12px;
            font-size: 16px;
            border-radius: 8px;
            border: 1px solid #ddd;
            width: calc(100% - 24px);
            box-sizing: border-box;
            margin: 5px 0;
        }
        input:focus, select:focus {
            border-color: #3498db;
            outline: none;
        }
        .button {
            display: inline-block;
            padding: 12px 24px;
            font-size: 18px;
            color: #fff;
            background-color: #3498db;
            text-decoration: none;
            border-radius: 8px;
            margin-top: 20px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.3s;
        }
        .button:hover {
            background-color: #2980b9;
            transform: scale(1.05);
        }
        .results-container {
            padding: 20px;
            margin-top: 20px;
            border-top: 1px solid #ddd;
        }
        .results {
            display: flex;
            flex-direction: column;
        }
        .product {
            border-bottom: 1px solid #ddd;
            padding: 10px 0;
            display: flex;
            align-items: center;
        }
        .product img {
            max-width: 100px;
            height: auto;
            margin-right: 20px;
        }
        .product-info {
            display: flex;
            flex-direction: column;
        }
        .product-info a {
            color: #3498db;
            text-decoration: none;
        }
        .product-info a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Zepto Product Search</h1>
        <p>Find products from our catalog by entering your search terms and applying filters.</p>
        <form id="searchForm" method="get">
            <div class="form-group">
                <label for="query">Search Term:</label>
                <input type="text" id="query" name="query" required>
            </div>
            <div class="form-group">
                <label for="max_price">Max Price:</label>
                <input type="number" id="max_price" name="max_price" step="0.01" min="0">
            </div>
            <div class="form-group">
                <label for="min_discount">Min Discount (%):</label>
                <input type="number" id="min_discount" name="min_discount" step="0.01" min="0">
            </div>
            <div class="form-group">
                <label for="min_rating">Min Rating:</label>
                <input type="number" id="min_rating" name="min_rating" step="0.1" min="0" max="5">
            </div>
            <button type="submit" class="button">Search</button>
        </form>
    </div>
        <div id="results-container" class="results-container" style="margin:20px">

            <div id="results" class="results">
                <!-- Search results will be inserted here -->
            </div>
        </div>
                        
    <script>
        document.getElementById('searchForm').addEventListener('submit', async function(event) {
            event.preventDefault();  // Prevent form from submitting the traditional way

            const query = document.getElementById('query').value;
            const maxPrice = document.getElementById('max_price').value || 99999;  // Default to a large value
            const minDiscount = document.getElementById('min_discount').value || 0;  // Default to 0%
            const minRating = document.getElementById('min_rating').value || 0;  // Default to 0

            const response = await fetch(`/search?query=${encodeURIComponent(query)}&max_price=${maxPrice}&min_discount=${minDiscount}&min_rating=${minRating}`);
            const products = await response.json();

            const resultsContainer = document.getElementById('results');
            resultsContainer.innerHTML = '';  // Clear previous results

            if (products.length > 0) {
                products.forEach(product => {
                    const productElement = document.createElement('div');
                    productElement.className = 'product';
                    productElement.innerHTML = `
                        <img src="${product.image}" alt="${product.name}">
                        <div class="product-info">
                            <a href="${product.url}" target="_blank">${product.name}</a><br>
                            Price: ₹${product.price}
                        </div>
                    `;
                    resultsContainer.appendChild(productElement);
                });
            } else {
                resultsContainer.innerHTML = '<p>No results found.</p>';
            }
        });
    </script>
</body>
</html>


    """)
    
@app.get("/search")
async def search(query: str, max_price: float = Query(99999), min_discount: float = Query(0), min_rating: float = Query(0)):
    products = get_relevant_products(query, max_price, min_discount, min_rating)
    return products
