# Projet : Systeme de recommandation d'images

## Presentation

Dans ce projet, vous allez construire un **systeme de recommandation d'images** qui suggere des images aux utilisateurs en fonction de leurs preferences. Ce projet met en pratique toutes les competences acquises lors des seances pratiques : analyse de donnees, visualisation, regroupement, classification et apprentissage automatique.

**Duree** : 3 seances pratiques
**Taille de l'equipe** : 2-3 etudiants
**Livrables** :
1. Un notebook Jupyter (`Nom1_Nom2_[Nom3].ipynb`)
2. Un rapport de synthese de 4 pages (PDF)

---

## Objectifs d'apprentissage

En completant ce projet, vous serez capable de :
- Automatiser la collecte de donnees a partir de sources web
- Extraire et traiter les metadonnees d'images
- Appliquer des algorithmes de regroupement pour analyser les caracteristiques des images
- Construire des profils de preferences utilisateur
- Implementer un algorithme de recommandation
- Visualiser efficacement les donnees
- Ecrire des tests complets pour votre systeme

---

## Architecture du projet

Le systeme est compose de 7 taches interconnectees :

```
┌─────────────────────────────────────────────────────────────────┐
│               SYSTEME DE RECOMMANDATION D'IMAGES                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 1. Collecte  │───▶│ 2. Etiquetage│───▶│ 3. Analyse   │       │
│  │ de donnees   │    │ & Annotation │    │ de donnees   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│        │                    │                   │                │
│        ▼                    ▼                   ▼                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │          Fichiers JSON (Stockage des metadonnees)     │       │
│  └──────────────────────────────────────────────────────┘       │
│        │                    │                   │                │
│        ▼                    ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 4. Visuali-  │    │ 5. Systeme de│    │ 6. Tests     │       │
│  │ sation       │    │ recommandat. │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                             │                                    │
│                             ▼                                    │
│                    ┌──────────────┐                              │
│                    │ 7. Rapport   │                              │
│                    │ de synthese  │                              │
│                    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

![Architecture](../../images/Project-Architecture.png "Architecture")

---

## Projet partie 1

---

### Tache 1 : Collecte de donnees

#### Objectif
Collecter au moins **100 images sous licence ouverte** avec leurs metadonnees.

#### Ce que vous devez faire

1. **Creer une structure de dossiers** :
   ```
   projet/
   ├── images/           # Images telechargees
   ├── data/             # Fichiers de metadonnees JSON
   └── projet.ipynb      # Votre notebook
   ```

2. **Trouver des sources d'images** (une ou plusieurs) :
   - [Wikimedia Commons](https://commons.wikimedia.org/) - Utilisez des requetes SPARQL (comme dans le TP 1)
   - [Unsplash API](https://unsplash.com/developers) - API gratuite pour des images de haute qualite
   - [Pexels API](https://www.pexels.com/api/) - Photos libres de droits
   - [Flickr API](https://www.flickr.com/services/api/) - Images Creative Commons

3. **Telecharger les images par programme** en utilisant les techniques du TP 1, Exercice 6

4. **Extraire et sauvegarder les metadonnees** de chaque image :
   - Nom du fichier image
   - Dimensions de l'image (largeur, hauteur)
   - Format du fichier (.jpg, .png, etc.)
   - Taille du fichier (en Ko)
   - URL source
   - Informations de licence
   - Donnees EXIF (si disponibles) : modele d'appareil photo, date de prise de vue, etc.

#### Resultat attendu
- Dossier `images/` contenant 100+ images
- `data/images_metadata.json` contenant les metadonnees de toutes les images

#### Conseils
- Utilisez `PIL` pour obtenir les dimensions de l'image
- Utilisez `os.path.getsize()` pour obtenir la taille du fichier
- Utilisez l'extraction EXIF (voir TP 2, Exercice 2)
- Stockez les metadonnees sous forme de liste de dictionnaires au format JSON

---

### Tache 2 : Etiquetage et annotation

#### Objectif
Ajouter des etiquettes descriptives et des caracteristiques calculees a chaque image.

#### Ce que vous devez faire

1. **Extraire les informations de couleur** en utilisant le regroupement KMeans (TP 2, Exercice 3) :
   - Trouver les 3 a 5 couleurs predominantes dans chaque image
   - Stocker les couleurs sous forme de valeurs RGB et/ou de noms de couleurs

2. **Determiner l'orientation de l'image** :
   - Paysage (largeur > hauteur)
   - Portrait (hauteur > largeur)
   - Carre (largeur ≈ hauteur)

3. **Ajouter des tags de categorie** (choisir une approche) :
   - **Manuelle** : Creer une interface simple pour tagger les images
   - **Automatisee** : Utiliser les categories/tags de la source d'images
   - **Hybride** : Commencer avec les tags sources, permettre le raffinement par l'utilisateur

4. **Classifier la taille de l'image** :
   - Vignette : < 500px
   - Moyenne : 500-1500px
   - Grande : > 1500px

#### Resultat attendu
- `data/images_labels.json` avec les metadonnees enrichies :
```json
{
  "image_001.jpg": {
    "predominant_colors": [[255, 128, 0], [0, 100, 200], [50, 50, 50]],
    "color_names": ["orange", "bleu", "gris"],
    "orientation": "paysage",
    "size_category": "moyenne",
    "tags": ["nature", "coucher de soleil", "plage"]
  }
}
```

#### Conseils
- Reutilisez votre code d'extraction de couleurs KMeans du TP 2
- Envisagez d'utiliser un mappage de noms de couleurs (RGB → nom de couleur)
- Stockez toutes les annotations dans un fichier JSON structure

---

### Tache 3 : Analyse de donnees

#### Objectif
Construire des profils de preferences utilisateur bases sur leurs selections d'images.

#### Ce que vous devez faire

1. **Simuler des utilisateurs** (creer au moins 5 utilisateurs) :
   - Chaque utilisateur met en "favori" 10 a 20 images
   - Les utilisateurs doivent avoir des preferences differentes (l'un aime la nature, un autre l'architecture, etc.)

2. **Construire les profils utilisateur** en analysant leurs images favorites :
   ```python
   user_profile = {
       "user_id": "user_001",
       "favorite_colors": ["bleu", "vert"],         # Couleurs les plus frequentes
       "favorite_orientation": "paysage",            # Orientation la plus frequente
       "favorite_size": "moyenne",                   # Taille la plus frequente
       "favorite_tags": ["nature", "eau"],            # Tags les plus frequents
       "favorite_images": ["img_01.jpg", ...]        # Liste des images favorites
   }
   ```

3. **Analyser les tendances** entre les utilisateurs :
   - Quelles couleurs sont les plus populaires globalement ?
   - Quels tags apparaissent le plus frequemment ?
   - Y a-t-il des groupes d'utilisateurs avec des preferences similaires ?

#### Resultat attendu
- `data/users.json` avec les profils utilisateur
- Resultats d'analyse montrant les tendances de preferences utilisateur

#### Conseils
- Utilisez pandas pour l'analyse de donnees (groupby, value_counts)
- Utilisez Counter du module collections pour trouver les elements les plus frequents
- Envisagez d'utiliser le regroupement pour grouper les utilisateurs similaires

---

### Tache 4 : Visualisation des donnees

#### Objectif
Creer des visualisations qui revelent des informations sur votre collection d'images et les preferences des utilisateurs.

#### Visualisations requises

1. **Statistiques de la collection d'images** :
   - Diagramme en barres : Nombre d'images par orientation
   - Diagramme en barres : Nombre d'images par categorie de taille
   - Diagramme circulaire : Distribution des formats d'image

2. **Analyse des couleurs** :
   - Afficher les couleurs predominantes sous forme de palettes
   - Histogramme des frequences de couleurs sur toutes les images

3. **Preferences des utilisateurs** :
   - Diagramme en barres : Couleurs favorites par utilisateur
   - Graphique comparatif : Preferences des utilisateurs cote a cote

4. **Analyse des tags** :
   - Diagramme en barres : Tags les plus courants
   - Nuage de mots (optionnel) : Visualisation de la frequence des tags

#### Resultat attendu
- Au moins 6 visualisations differentes dans votre notebook
- Tous les graphiques doivent avoir des titres, des labels et des legendes

#### Conseils
- Utilisez matplotlib pour toutes les visualisations (TP 2, Exercice 1)
- Sauvegardez les graphiques importants avec `plt.savefig()`
- Utilisez les sous-graphiques pour grouper les visualisations liees

---

### Tache 5 : Systeme de recommandation

#### Objectif
Implementer un systeme qui recommande des images aux utilisateurs en fonction de leurs preferences.

#### Choisissez votre approche

Vous devez implementer **au moins une** de ces approches :

##### Option A : Filtrage base sur le contenu (utilisant la classification)
Recommander des images similaires a celles que l'utilisateur a aimees.

```python
# Entrainer un classifieur sur les favoris de l'utilisateur
# Caracteristiques : couleur, orientation, taille, tags
# Label : Favori / Non Favori
# Predire quelles images non vues l'utilisateur aimerait
```

**Utiliser** : Arbre de decision, Foret aleatoire ou SVM (TP 3, Exercices 2-3)

##### Option B : Recommandation basee sur le regroupement
Grouper les images similaires et recommander celles du meme groupe.

```python
# Regrouper toutes les images selon leurs caracteristiques
# Trouver a quel groupe appartiennent les favoris de l'utilisateur
# Recommander d'autres images du meme groupe
```

**Utiliser** : KMeans (TP 2, Exercices 3-5)

##### Option C : Approche hybride
Combiner les deux methodes pour de meilleures recommandations.

#### Exigences d'implementation

1. **Entree** : Identifiant utilisateur
2. **Sortie** : Liste de 5 a 10 images recommandees (pas deja en favoris)
3. **Explication** : Breve raison pour laquelle chaque image est recommandee

#### Resultat attendu
```python
def recommend_images(user_id, n_recommendations=5):
    """
    Recommander des images pour un utilisateur.

    Args:
        user_id: L'utilisateur pour lequel recommander
        n_recommendations: Nombre d'images a recommander

    Returns:
        Liste de tuples (nom_fichier_image, raison)
    """
    # Votre implementation
    pass
```

#### Conseils
- Commencez avec les exemples dans `examples/recommendation.ipynb`
- Utilisez LabelEncoder pour convertir les caracteristiques categorielles en nombres
- Testez vos recommandations manuellement - sont-elles pertinentes ?

---

### Tache 6 : Tests

#### Objectif
Verifier que votre systeme fonctionne correctement.

#### Tests requis

1. **Tests de validation des donnees** :
   - Toutes les images existent dans le dossier images
   - Toutes les images ont des metadonnees
   - Les valeurs des metadonnees sont valides (pas de dimensions negatives, etc.)

2. **Tests fonctionnels** :
   - L'extraction de couleurs retourne des valeurs RGB valides
   - La generation de profils utilisateur fonctionne correctement
   - La fonction de recommandation retourne le nombre attendu de resultats

3. **Tests de qualite des recommandations** :
   - Les images recommandees ne sont pas deja dans les favoris de l'utilisateur
   - Les recommandations correspondent aux preferences de l'utilisateur (par exemple, si l'utilisateur aime les images bleues, les recommandations devraient inclure des images bleues)

#### Resultat attendu
```python
def test_data_integrity():
    """Tester que toutes les donnees sont valides"""
    # Vos tests
    assert len(images) >= 100, "Il faut au moins 100 images"
    assert all_images_have_metadata(), "Metadonnees manquantes"

def test_recommendation_system():
    """Tester que les recommandations fonctionnent"""
    recommendations = recommend_images("user_001", 5)
    assert len(recommendations) == 5, "Devrait retourner 5 recommandations"
    # Plus de tests...
```

#### Conseils
- Utilisez les instructions `assert` pour des tests simples
- Affichez des messages clairs de succes/echec
- Testez les cas limites (profil utilisateur vide, nouvel utilisateur, etc.)

---

### Tache 7 : Rapport de synthese

#### Objectif
Rediger un rapport de 4 pages resumant votre projet.

#### Sections requises

1. **Introduction** (0,5 page)
   - Objectif du projet
   - Votre approche en bref

2. **Collecte de donnees** (0,5 page)
   - Sources des images et licences
   - Nombre d'images collectees
   - Metadonnees stockees

3. **Methodologie** (1,5 pages)
   - Approche d'etiquetage (comment vous avez extrait les caracteristiques)
   - Construction du profil utilisateur
   - Algorithme de recommandation choisi et pourquoi
   - Inclure un diagramme d'architecture

4. **Resultats** (1 page)
   - Visualisations principales (2-3 figures)
   - Qualite/precision des recommandations
   - Observations interessantes

5. **Limites et travaux futurs** (0,25 page)
   - Qu'est-ce qui n'a pas bien fonctionne ?
   - Comment pourrait-on ameliorer le systeme ?

6. **Conclusion** (0,25 page)
   - Resume des realisations
   - Auto-evaluation

#### Format
- 4 pages maximum
- Format PDF
- Pas de code dans le rapport (uniquement des resultats et des explications)
- Inclure des references/bibliographie

---

### Evaluation (Partie 1)

| Tache | Points | Criteres principaux |
|-------|--------|---------------------|
| Collecte de donnees | 15% | Automatisation, 100+ images, metadonnees completes |
| Etiquetage et annotation | 15% | Extraction de couleurs, categorisation appropriee |
| Analyse de donnees | 15% | Profils utilisateur, analyse des preferences |
| Visualisation des donnees | 15% | 6+ visualisations, formatage correct |
| Systeme de recommandation | 20% | Algorithme fonctionnel, recommandations pertinentes |
| Tests | 10% | Tests complets, tous reussis |
| Rapport de synthese | 10% | Clair, complet, bien structure |

---

### Soumission (Partie 1)

#### Fichiers a soumettre
```
Nom1_Nom2_[Nom3].zip
├── Nom1_Nom2_[Nom3].ipynb    # Votre notebook
├── data/
│   ├── images_metadata.json
│   ├── images_labels.json
│   └── users.json
└── rapport_synthese.pdf
```

#### Notes importantes
- **NE PAS** soumettre le dossier images (trop volumineux)
- Assurez-vous que votre notebook s'execute sans erreurs
- Incluez des commentaires expliquant votre code
- Renommez les fichiers avec les noms des membres de votre equipe

---

### Pour commencer

1. Commencez avec le notebook modele : `fr/Projet/projet.ipynb`
2. Consultez les exemples dans `examples/recommendation.ipynb`
3. Reutilisez le code de vos seances pratiques
4. Travaillez de maniere incrementale - completez chaque tache avant de passer a la suivante

**Remarque** : Vous pouvez consulter des [exemples supplementaires](../../examples) de notebooks.

---

## Projet partie 2

L'objectif de la partie 2 est de transformer votre projet de la partie 1 en une **application distribuee et conteneurisee** utilisant Docker (TP 6) et PySpark (TP 5). Vous allez decomposer votre notebook monolithique en services independants et scalables qui communiquent via des volumes de donnees partages.

### Vue d'ensemble

Dans la partie 1, l'ensemble de votre pipeline (collecte de donnees, etiquetage, analyse, visualisation, recommandation) s'execute dans un seul notebook Jupyter. Dans la partie 2, vous allez :

1. **Identifier les taches independantes** de votre pipeline de la partie 1
2. **Remplacer les boucles sequentielles par des operations PySpark** (map-reduce, expressions lambda)
3. **Empaqueter chaque tache dans un conteneur Docker** avec son propre Dockerfile
4. **Utiliser des volumes Docker** pour partager les donnees (JSON, CSV, images) entre les conteneurs
5. **Orchestrer le tout** avec `docker-compose`

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        docker-compose.yml                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────────┐   │
│  │  Conteneur 1 :    │   │  Conteneur 2 :    │   │  Conteneur 3 : │   │
│  │  Acquisition      │──▶│  Analyse          │──▶│ Recommandation │   │
│  │  de donnees       │   │  & Etiquetage     │   │                │   │
│  │  - Telecharger    │   │  - Extraction     │   │ - Profil util. │   │
│  │  - Extraire EXIF  │   │  - Regroupement   │   │ - Recommander  │   │
│  │  - Sauv. metadon. │   │  - Profils util.  │   │ - Visualiser   │   │
│  └────────┬──────────┘   └────────┬──────────┘   └───────┬────────┘   │  v n
│           │                       │                       │            │
│           ▼                       ▼                       ▼            │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │                    Volume Docker partage                      │     │
│  │  /shared_data/                                                │     │
│  │  ├── images/               # Images telechargees              │     │
│  │  ├── images_metadata.json  # Du conteneur 1                   │     │
│  │  ├── images_labels.json    # Du conteneur 2                   │     │
│  │  ├── recommandation/                                       │     │
│  │  │  ├── users.json           # Du conteneur 3                 │     │
│  │  │  └── recommendations.json # Du conteneur 3                 │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Etape 1 : Identifier les taches independantes

Examinez votre notebook de la partie 1 et decomposez-le en au moins **trois taches independantes**. Decomposition suggeree :

| Conteneur | Taches de la partie 1 couvertes | Entree | Sortie |
|-----------|--------------------------------|--------|--------|
| **Acquisition de donnees** | Tache 1 (Collecte de donnees) | URLs des sources d'images | `images/`, `images_metadata.json` |
| **Analyse & Etiquetage** | Tache 2 (Etiquetage), Tache 3 (Analyse), Tache 4 (Visualisation) | `images/`, `images_metadata.json` | `images_labels.json`, `users.json`, visualisations PNG |
| **Recommandation** | Tache 5 (Recommandation), Tache 6 (Tests) | `images_labels.json`, `users.json` | `/shared_data/recommandation/recommendations.json`, resultats de tests |

Vous pouvez decomposer davantage en plus de conteneurs (par exemple, des conteneurs separes pour l'etiquetage, l'analyse et la visualisation).

### Etape 2 : Remplacer les boucles par des operations PySpark

Pour chaque conteneur, identifiez les **boucles sequentielles** de votre code de la partie 1 et reecrivez-les en utilisant les operations distribuees de PySpark :

#### Exemple : Extraction de couleurs (de la tache 2)

**Partie 1 (boucle sequentielle) :**
```python
# Traitement des images une par une dans une boucle
results = []
for image_file in image_files:
    img = load_image(image_file)
    colors = extract_colors(img)
    results.append({"file": image_file, "colors": colors})
```

**Partie 2 (PySpark) :**
```python
from pyspark import SparkContext

sc = SparkContext("local[*]", "ExtractionCouleurs")

# Distribuer la liste des fichiers images sur les workers
images_rdd = sc.parallelize(image_files)

# Map : appliquer l'extraction de couleurs a chaque image en parallele
colors_rdd = images_rdd.map(lambda f: (f, extract_colors(load_image(f))))

# Collecter les resultats
results = colors_rdd.collect()
```

#### Exemple : Construction du profil utilisateur (de la tache 3)

**Partie 1 (boucle sequentielle) :**
```python
# Construction des profils un utilisateur a la fois
profiles = []
for user in users:
    fav_colors = []
    for img in user["favorites"]:
        fav_colors.extend(labels[img]["color_names"])
    most_common = Counter(fav_colors).most_common(3)
    profiles.append({"user": user["id"], "colors": most_common})
```

**Partie 2 (PySpark) :**
```python
# Distribuer le traitement des utilisateurs avec les RDD
users_rdd = sc.parallelize(users)

# Map chaque utilisateur vers son profil en utilisant lambda
profiles_rdd = users_rdd.map(lambda u: build_user_profile(u, labels))

# Utiliser reduceByKey pour agreger les comptages de tags de tous les utilisateurs
all_tags_rdd = users_rdd.flatMap(lambda u: get_user_tags(u, labels)) \
                        .map(lambda tag: (tag, 1)) \
                        .reduceByKey(lambda a, b: a + b)

top_tags = all_tags_rdd.sortBy(lambda x: x[1], ascending=False).take(10)
```

#### Exemple : Calcul de score de recommandation (de la tache 5)

**Partie 1 (boucle sequentielle) :**
```python
# Notation des images pour un utilisateur de maniere sequentielle
scores = []
for img in all_images:
    score = compute_similarity(user_profile, image_features[img])
    scores.append((img, score))
scores.sort(key=lambda x: x[1], reverse=True)
```

**Partie 2 (PySpark) :**
```python
# Distribuer le calcul de scores sur les workers
images_rdd = sc.parallelize(all_images)

# Map : calculer le score de similarite en parallele
scores_rdd = images_rdd.map(lambda img: (img, compute_similarity(user_profile, image_features[img])))

# Trier et prendre les meilleures recommandations
recommendations = scores_rdd.sortBy(lambda x: x[1], ascending=False).take(10)
```

### Etape 3 : Creer les conteneurs Docker

Chaque conteneur necessite :
- **Dockerfile** avec les dependances requises (Python, PySpark, bibliotheques)
- **Script(s) Python** implementant la tache avec les operations PySpark
- **requirements.txt** listant les dependances Python

Referez-vous aux exemples du TP 6 pour la structure des conteneurs :
- Exemple `SharedVolume/` pour la communication via volumes entre conteneurs
- Exemple `AppDB/` pour l'orchestration multi-conteneurs

#### Exemple de Dockerfile pour le conteneur d'analyse de donnees :

```dockerfile
FROM python:3.10-slim

# Installer Java (requis pour PySpark)
RUN apt-get update && apt-get install -y openjdk-17-jdk-headless && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analyze.py .
CMD ["python", "analyze.py"]
```

#### Exemple de requirements.txt :
```
pyspark==3.5.3
numpy==1.26.4
Pillow==10.4.0
```

### Etape 4 : Utiliser les volumes Docker pour le partage de donnees

Utilisez un volume Docker partage pour que les conteneurs puissent lire et ecrire des fichiers de donnees (JSON, CSV, images). Chaque conteneur lit ses entrees depuis le volume partage et ecrit ses sorties dans le volume partage.

#### Exemple de docker-compose.yml :

```yaml
version: "3.8"

services:
  acquisition:
    build: ./acquisition
    volumes:
      - shared_data:/shared_data
    environment:
      - OUTPUT_DIR=/shared_data

  analysis:
    build: ./analysis
    depends_on:
      acquisition:
        condition: service_completed_successfully
    volumes:
      - shared_data:/shared_data
    environment:
      - INPUT_DIR=/shared_data
      - OUTPUT_DIR=/shared_data

   recommendation:
      build: ./recommandation
    depends_on:
      analysis:
        condition: service_completed_successfully
    volumes:
      - shared_data:/shared_data
    environment:
      - INPUT_DIR=/shared_data
      - OUTPUT_DIR=/shared_data/recommandation

volumes:
  shared_data:
```

### Etape 5 : Executer et verifier

```bash
# Construire et executer tous les conteneurs en sequence
docker compose up --build

# Verifier le volume partage pour les sorties
docker compose run --rm recommendation ls /shared_data/recommandation/
```

### Evaluation (Partie 2)

| Criteres | Points | Details |
|----------|--------|---------|
| Decomposition en taches | 25% | Separation claire en 3+ conteneurs independants, chacun avec des entrees/sorties bien definies |
| Volumes Docker | 25% | Utilisation correcte des volumes partages pour passer les fichiers JSON, CSV et images entre conteneurs |
| Map-reduce et expressions lambda | 25% | Boucles sequentielles de la partie 1 remplacees par des operations PySpark `map`, `flatMap`, `filter`, `reduceByKey` et des expressions lambda |
| Utilisation de PySpark | 25% | Utilisation correcte de SparkContext, operations RDD et traitement distribue dans au moins 2 conteneurs |

**Remarque** : Vous pouvez consulter des [exemples supplementaires](../../examples) de conteneurs Docker ainsi que les exemples du TP 5 et du TP 6.
