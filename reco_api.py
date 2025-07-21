from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import re

# =========================================================================
# Configuration du répertoire de cache de TensorFlow Hub (pour le modèle)
os.environ['TFHUB_CACHE_DIR'] = 'C:\\Users\\hp\\tensorflow_datasets'
# =========================================================================

app = FastAPI()

# =========================================================================
# ANCIEN CODE: Servir les images statiques via FastAPI n'est plus nécessaire si les URLs viennent de la DB
# app.mount("/images", StaticFiles(directory=STATIC_IMAGES_DIR), name="images")
# =========================================================================

# ========================== Chargement du modèle ==========================
model = hub.KerasLayer("https://tfhub.dev/google/experts/bit/r50x1/in21k/consumer_goods/1")

# ========================== Fonctions de traitement ==========================
def load_image(path):
    image_data = tf.io.read_file(path)
    image = tf.image.decode_image(image_data, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def resize_images(image):
    return tf.image.resize(image, (512, 512), preserve_aspect_ratio=True)

def l2_normalize(feature_vector):
    return tf.math.l2_normalize(feature_vector, axis=1)

def extract_features(image_paths, model, batch_size=32):
    db = tf.data.Dataset.from_tensor_slices(list(map(str, image_paths)))
    db = db.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    db = db.map(resize_images, num_parallel_calls=tf.data.AUTOTUNE)
    db = db.batch(batch_size)
    db = db.prefetch(tf.data.AUTOTUNE)

    features = db.map(model)
    features = features.map(l2_normalize)

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    features_list = []
    for feature_batch in tqdm(features, total=total_batches, desc="Extraction par lots"):
        features_list.append(feature_batch.numpy())
    
    features = np.concatenate(features_list)
    return features

def unpad_image(image_np):
    image_np = np.stack([image_np, image_np, image_np], axis=-1) if image_np.ndim < 3 else image_np
    rr, cc = np.nonzero((image_np != 255).any(axis=2))
    y0, y1 = rr.min(), rr.max()
    x0, x1 = cc.min(), cc.max()
    return image_np[y0:y1+1, x0:x1+1]

def show_images(paths, figsize=None):
    n_images = len(paths)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    for ax, path in zip(axes, paths):
        image_np = plt.imread(path)
        image_np = unpad_image(image_np)
        image_np = resize(image_np, (363, 266))
        ax.imshow(image_np)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig

# Helper function to extract product ID from a path/filename
def extract_product_id_from_path(input_string):
    filename = Path(input_string).name 
    match = re.match(r'^(\d+)\.jpg$', filename)
    if match:
        return match.group(1)
    return None

# ========================== Prétraitement de la base ==========================
image_base_dir = Path('C:/Users/hp/Downloads/data')
db_image_paths = sorted([str(p) for p in image_base_dir.rglob('*.jpg')])

# Définir le chemin pour sauvegarder/charger les caractéristiques extraites des images
FEATURES_CACHE_PATH = Path('C:/Users/hp/tensorflow_datasets/image_features.npy') 

if FEATURES_CACHE_PATH.exists():
    print(f"Chargement des caractéristiques d'images depuis {FEATURES_CACHE_PATH}...")
    image_features = np.load(FEATURES_CACHE_PATH)
    print("Caractéristiques chargées avec succès.")
else:
    print("Extraction des caractéristiques d'images (cela peut prendre du temps)...")
    image_features = extract_features(db_image_paths, model, batch_size=64)
    
    FEATURES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_CACHE_PATH, image_features)
    print(f"Caractéristiques extraites et sauvegardées dans {FEATURES_CACHE_PATH}.")

# ========================== Endpoint API ==========================
@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    try:
        query_product_id = extract_product_id_from_path(file.filename)

        image = Image.open(file.file).convert("RGB")
        image = image.resize((512, 512))
        image_np = np.array(image) / 255.0
        image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0) 

        feature = model(image_tensor)
        feature = l2_normalize(feature).numpy()

        similarities = cosine_similarity(feature, image_features)[0]
        
        indexed_similarities = []
        for i, s in enumerate(similarities):
            indexed_similarities.append((s, i))

        indexed_similarities.sort(key=lambda x: x[0], reverse=True)

        recommended_product_ids = [] # MODIFIÉ : Nous allons renvoyer des IDs
        num_found = 0
        NUM_RECOMMENDATIONS = 10 
        
        for sim, original_idx in indexed_similarities:
            recommended_product_path = db_image_paths[original_idx]
            recommended_product_id = extract_product_id_from_path(recommended_product_path)

            if query_product_id and recommended_product_id == query_product_id:
                continue 
            
            # MODIFIÉ : Ajouter l'ID du produit recommandé à la liste
            if recommended_product_id: # Assurez-vous que l'ID a pu être extrait
                recommended_product_ids.append(recommended_product_id)
                num_found += 1
                if num_found >= NUM_RECOMMENDATIONS:
                    break
        
        # MODIFIÉ : Renvoie la liste des IDs de produits
        return {"results": recommended_product_ids}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})