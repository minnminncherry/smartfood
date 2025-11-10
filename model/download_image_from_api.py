import os
import requests
import json
import torch.nn.functional as F
import time

# Global Variable Decleartion
CATEGORIES = ["breakfast-cereals","popcorn","waters"]
LIMIT = 50
OUTPUT_DIR = "food_images"

def create_output_dirs(categories, output_dir):
    for category in categories:
        images_path = os.path.join(output_dir, category, "images")
        labels_path = os.path.join(output_dir, category, "labels")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

def download_images_from_api(categories, limit, output_dir):
    for j in categories:
        print("catrgory_name : ", j)
        url = f"https://world.openfoodfacts.org/category/{j}.json?page_size={limit}"
        print ("URL : ", url)
        
        response = requests.get(url)
        data = response.json()
        products = data.get("products", [])
        
        for i, product in enumerate(products):
            name = product.get("product_name", f"unknown_{i}").replace("/", "_")
            image_url = product.get("image_front_url")
        
            # Skip if no image
            if not image_url:
                continue
        
            # Download image
            try:
                img_data = requests.get(image_url, timeout=10).content
                img_path = os.path.join(f"{output_dir}/{j}", "images", f"{name}.jpg")
                with open(img_path, "wb") as handler:
                    handler.write(img_data)
        
                # 1. Product Category (from API)
                product_category = j
        
                # 2. Meal Type (based on category keywords)
                if any(k in categories for k in ["banana", "apple", "blueberry"]):
                    meal_type = "fruit"
                elif any(k in categories for k in ["lunch", "dinner", "rice", "noodle", "pasta"]):
                    meal_type = "meal"
                elif any(k in categories for k in ["water", "beverage"]):
                    meal_type = "drink"
                else:
                    meal_type = "snack"
        
                # 3. Gluten Free (based on tags/ingredients)
                gluten_free = False
                if "gluten-free" in str(product.get("labels_tags", [])) or "gluten-free" in str(product.get("ingredients_text", "")).lower():
                    gluten_free = True
        
                label_data = {
                    "product_name": name,
                    "brands": product.get("brands"),
                    "product_category": product_category,
                    "meal_type": meal_type,
                    "gluten_free": gluten_free,
                    "nutriments": product.get("nutriments", {}),
                    "image_path": img_path
                }
        
                label_path = os.path.join(f"{output_dir}/{j}", "labels", f"{name}.json")
                with open(label_path, "w") as label_file:
                    json.dump(label_data, label_file, indent=4)
        
                print(f"[{i+1}] Downloaded: {name}")
        
                # Wait a bit to be kind to the server
                time.sleep(0.5)
        
            except Exception as e:
                print(f"Failed to download {name}: {e}")