# 🧠 FastAPI Recommendation Service

Ce dépôt contient le service de recommandation d'images pour l'application **Fashion Store API**. Développé avec **FastAPI** en Python, ce service est spécialisé dans l'extraction de caractéristiques d'images et la recherche de similarités pour recommander des produits de mode. Il fonctionne comme un microservice distinct, appelé par le backend Spring Boot.

## 🚀 Vue d'ensemble de l'Architecture

Ce service est une API REST légère qui expose un endpoint pour la recommandation de produits. Il est conçu pour être déployé et exécuté indépendamment du backend Spring Boot.

* **Rôle :** Reçoit une image, traite celle-ci à l'aide d'un modèle de Machine Learning pré-entraîné, calcule la similarité avec une base de données d'images de produits, et retourne les identifiants des produits les plus similaires.

* **Communication :** Interagit via HTTP avec le backend Spring Boot, qui lui transmet les images de requête et utilise les IDs de produits retournés.

## 📦 Fonctionnalités

* 📸 **Extraction de caractéristiques d'images :** Utilise un modèle **TensorFlow Hub** (`google/experts/bit/r50x1/in21k/consumer_goods/1`) pour convertir les images en vecteurs numériques (caractéristiques).

* 🗄️ **Base de données d'images :** Charge et pré-traite une collection locale d'images de produits pour créer une base de données de caractéristiques.

* ⚡ **Mise en cache des caractéristiques :** Sauvegarde les caractéristiques extraites sur disque (`image_features.npy`) pour un rechargement rapide et éviter de ré-extraire les features à chaque démarrage.

* 🔍 **Recherche de similarité :** Calcule la similarité cosinus entre l'image de requête et les caractéristiques de la base de données.

* 💡 **Recommandation :** Retourne les identifiants des `K` produits les plus similaires à l'image fournie.

## 🧱 Technologies utilisées

* **Python 3.10+**

* **FastAPI :** Framework web pour construire des API.

* **Uvicorn :** Serveur ASGI pour exécuter l'application FastAPI.

* **Pillow :** Bibliothèque pour le traitement d'images.

* **NumPy :** Pour les opérations numériques et les vecteurs de caractéristiques.

* **TensorFlow & TensorFlow Hub :** Pour le chargement et l'utilisation du modèle de Deep Learning (extraction de caractéristiques).

* **Scikit-learn :** Pour le calcul de la similarité cosinus.

* **Tqdm :** Pour afficher des barres de progression (utile lors de l'extraction initiale des caractéristiques).

* **Matplotlib & Scikit-image :** Utilisés dans le notebook d'exploration pour la visualisation des images et le prétraitement (non directement dans l'API de production, mais utiles pour la compréhension).

## 🛠️ Configuration et Démarrage

### 1. Structure du projet

Assurez-vous que ce dépôt (`fastapi-recommendation-service/`) est un dossier séparé de votre projet Spring Boot.

```bash
fastapi-recommendation-service/
├── reco_api.py                 # Le code principal de l'API FastAPI
├── requirements.txt            # Liste des dépendances Python
├── Fashion.ipynb               # Notebook Jupyter pour l'exploration et le développement du modèle
├── .gitignore                  # Fichier .gitignore spécifique à ce dépôt Python
└── ... (vos modèles ML si sauvegardés localement, autres scripts, etc.)
```

### 2. Préparer les données et le cache des modèles

Le service FastAPI a besoin d'accéder à vos images de produits et d'un répertoire pour mettre en cache les modèles TensorFlow Hub et les caractéristiques extraites.

* **Répertoire des images de produits :**
    Le code est configuré pour chercher les images dans :
    `C:\Users\hp\Downloads\data`
    **Assurez-vous que ce chemin existe sur votre système et qu'il contient toutes vos images de produits (par exemple, dans des sous-dossiers comme `Apparel/Boys/Images/images_with_product_ids/`).**

* **Répertoire de cache TensorFlow Hub et caractéristiques :**
    Le service utilisera :
    `C:\Users\hp\tensorflow_datasets`
    pour télécharger et mettre en cache le modèle TensorFlow Hub, et pour sauvegarder/charger le fichier `image_features.npy` (qui contient les caractéristiques extraites de toutes vos images).
    **Créez ce dossier si il n'existe pas.**

    ```python
    # Dans reco_api.py
    os.environ['TFHUB_CACHE_DIR'] = 'C:\\Users\\hp\\tensorflow_datasets'
    image_base_dir = Path('C:/Users/hp/Downloads/data')
    FEATURES_CACHE_PATH = Path('C:/Users/hp/tensorflow_datasets/image_features.npy')
    ```

    **Modifiez ces chemins dans `reco_api.py` si vos répertoires sont différents.**

### 3. Installation des dépendances Python

Il est **fortement recommandé** d'utiliser un [environnement virtuel Python](https://docs.python.org/3/library/venv.html) pour isoler les dépendances de votre projet et éviter les conflits avec d'autres projets Python sur votre système.

1.  **Naviguez** dans le dossier `fastapi-recommendation-service/` dans votre terminal :
    ```bash
    cd fastapi-recommendation-service/
    ```

2.  **Créez un environnement virtuel** (si vous n'en avez pas déjà un) :
    ```bash
    python -m venv venv
    ```
    Ceci créera un dossier `venv/` (ou `.venv/`) contenant un environnement Python isolé.

3.  **Activez l'environnement virtuel :**
    * **Sur Windows (CMD) :**
        ```bash
        venv\Scripts\activate.bat
        ```
    * **Sur Windows (PowerShell) :**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    * **Sur Linux/macOS ou Git Bash :**
        ```bash
        source venv/bin/activate
        ```
    Une fois activé, votre invite de commande devrait afficher `(venv)` au début, indiquant que vous êtes dans l'environnement isolé.

4.  **Installez les dépendances :**
    Assurez-vous d'avoir un fichier `requirements.txt` dans ce dossier avec le contenu suivant :
    ```
    fastapi
    uvicorn[standard]
    Pillow
    scikit-learn
    numpy
    tensorflow
    tensorflow-hub
    tqdm
    matplotlib
    scikit-image
    ```
    Puis installez-les :
    ```bash
    pip install -r requirements.txt
    ```
    Ceci installera toutes les bibliothèques nécessaires dans votre environnement virtuel.

### 4. 🚀 Démarrer le service FastAPI

Une fois les dépendances installées et l'environnement virtuel activé :

1.  **Assurez-vous d'être toujours dans le dossier `fastapi-recommendation-service/`**.
2.  **Lancez l'application Uvicorn :**
    ```bash
    uvicorn reco_api:app --host 0.0.0.0 --port 8000
    ```
    * `reco_api`: Fait référence à votre fichier `reco_api.py`.
    * `app`: Fait référence à l'instance `app = FastAPI()` définie dans votre `reco_api.py`.
    * `--host 0.0.0.0`: Rend le service accessible depuis l'extérieur de votre machine (utile pour les conteneurs ou si le backend Spring Boot est sur une autre machine).
    * `--port 8000`: Le port sur lequel le service FastAPI écoutera. Assurez-vous que ce port est libre et qu'il correspond à la configuration `fastapi.url` dans votre `application.properties` Spring Boot.

    Lors du premier démarrage, le service téléchargera le modèle TensorFlow Hub et extraira les caractéristiques de toutes vos images. Cela peut prendre un certain temps. Les caractéristiques seront ensuite sauvegardées dans `image_features.npy` pour des démarrages plus rapides.

---

## 🌐 Endpoint API

Ce service expose un seul endpoint principal, qui est consommé en interne par le backend Spring Boot.

* **Endpoint de recommandation :**
    ```
    POST http://localhost:8000/recommendations
    ```
    **Description :** Cet endpoint attend une requête `multipart/form-data` avec un champ de fichier nommé `image`. Il traite l'image, trouve les produits les plus similaires dans sa base de données et retourne une liste d'IDs de produits.

    **Exemple de réponse (succès) :**
    ```json
    {
      "results": [
        "product_id_123",
        "product_id_456",
        "product_id_789"
      ]
    }
    ```

---

## 📁 Structure du projet

```bash
fastapi-recommendation-service/
├── reco_api.py                 # Le code principal de l'API FastAPI
├── requirements.txt            # Liste des dépendances Python
├── Fashion.ipynb               # Notebook Jupyter pour l'exploration et le développement du modèle
├── .gitignore                  # Fichier .gitignore spécifique à ce dépôt Python
├── venv/                       # Dossier de l'environnement virtuel (ignoré par Git)
├── image_features.npy          # Fichier cache des caractéristiques d'images (généré, peut être ignoré si trop grand)
└── ... (vos modèles ML si sauvegardés localement, autres scripts, etc.)
```

---

## 📊 Notebook d'exploration (`Fashion.ipynb`)

Le fichier `Fashion.ipynb` est un Jupyter Notebook qui a été utilisé pour l'exploration initiale des données, le test du modèle de caractéristiques d'images, et la validation de la logique de similarité.

Il contient les étapes pour :
-   Charger les données `fashion.csv`.
-   Charger et utiliser le modèle `google/experts/bit/r50x1/in21k/consumer_goods/1` de TensorFlow Hub.
-   Extraire les caractéristiques des images.
-   Calculer la similarité cosinus.
-   Visualiser les résultats de recommandation.

Ce notebook est une excellente ressource pour comprendre le fonctionnement sous-jacent du modèle de recommandation.

---

💬 N’hésitez pas à cloner, tester et améliorer !
