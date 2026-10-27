# BigData-Project - Partie 2

Ce depot contient la partie 2 du projet: transformation du notebook initial en pipeline distribue et conteneurise avec Docker Compose et PySpark.

L'objectif est de separer le traitement en trois services independants:

- `acquisition`: collecte les images et leurs metadonnees.
- `analysis`: analyse les images, extrait les caracteristiques, cree les labels et les visualisations.
- `recommendation`: genere des profils utilisateurs, calcule les recommandations et produit les resultats de tests.

Les conteneurs communiquent uniquement avec des fichiers partages dans un volume Docker nomme, monte dans chaque conteneur sur `/shared_data`.

## Architecture

```text
.
|-- docker-compose.yml
|-- acquisition/
|   |-- Dockerfile
|   |-- requirements.txt
|   `-- acquire.py
|-- analysis/
|   |-- Dockerfile
|   |-- requirements.txt
|   `-- analyze.py
|-- recommandation/
|   |-- Dockerfile
|   |-- requirements.txt
|   `-- recommend.py
|-- Projet.md
|-- README.md
`-- RULLIAT_AOUCHICHE_MONEGAT.ipynb
```

Le volume Docker declare dans `docker-compose.yml` est:

```yaml
volumes:
  shared_data:
```

Il est monte dans les trois services:

```yaml
volumes:
  - shared_data:/shared_data
```

## Flux De Donnees

Le pipeline s'execute dans cet ordre:

1. `acquisition`
   - interroge Wikidata/Wikimedia avec SPARQL;
   - telecharge les images;
   - extrait les metadonnees locales;
   - ecrit `/shared_data/images/` et `/shared_data/images_metadata.json`.

2. `analysis`
   - lit `/shared_data/images/` et `/shared_data/images_metadata.json`;
   - extrait les caracteristiques visuelles;
   - calcule les couleurs dominantes, l'orientation, la taille, les tags;
   - applique un clustering KMeans avec PySpark;
   - ecrit `/shared_data/images_features.json`, `/shared_data/images_labels.json` et `/shared_data/visualizations/`.

3. `recommendation`
   - lit les features et labels produits par l'analyse;
   - genere des utilisateurs fictifs;
   - construit leurs profils;
   - calcule les recommandations;
   - ecrit `/shared_data/recommandation/users.json`, `/shared_data/recommandation/recommendations.json` et `/shared_data/recommandation/test_results.json`.

Docker Compose impose cet ordre avec `depends_on` et `condition: service_completed_successfully`.

## Utilisation De PySpark

PySpark est utilise dans deux conteneurs, comme demande dans la consigne.

Dans `analysis/analyze.py`:

- `sc.parallelize(...)` distribue les metadonnees;
- `map(...)` extrait les features image par image;
- `filter(...)` retire les images invalides;
- `flatMap(...)` extrait couleurs et tags;
- `reduceByKey(...)` agrege les frequences;
- `pyspark.ml.clustering.KMeans` effectue le regroupement.

Dans `recommandation/recommend.py`:

- `sc.parallelize(...)` distribue les images, labels et utilisateurs;
- `map(...)` genere les profils utilisateurs et recommandations;
- `flatMap(...)` extrait des listes de couleurs;
- `reduceByKey(...)` calcule les frequences globales.

Cela remplace une partie des boucles sequentielles du notebook par des operations distribuees.

## Lancer Le Projet

Depuis la racine du depot:

```powershell
docker compose up --build
```

Pour repartir de zero, en supprimant aussi le volume Docker contenant les anciens resultats:

```powershell
docker compose down -v
docker compose up --build
```

## Parametres

Les principaux parametres sont configurables avec des variables d'environnement.

Valeurs par defaut dans `docker-compose.yml`:

- `IMAGE_LIMIT=100`: nombre d'images a collecter.
- `CANDIDATE_FACTOR=1.5`: marge de candidats SPARQL pour compenser les echecs de telechargement.
- `DOWNLOAD_WORKERS=16`: nombre de telechargements paralleles.
- `CLUSTER_K=8`: nombre de clusters KMeans.
- `USER_LIMIT=1000`: nombre d'utilisateurs generes.
- `RECO_PER_USER=10`: nombre de recommandations par utilisateur.
- `SEED=42`: graine aleatoire.

Exemple PowerShell:

```powershell
$env:IMAGE_LIMIT="200"
$env:USER_LIMIT="1000"
$env:CLUSTER_K="8"
docker compose up --build
```

Exemple bash:

```bash
IMAGE_LIMIT=200 USER_LIMIT=1000 CLUSTER_K=8 docker compose up --build
```

## Lire Les Resultats

Les resultats ne sont pas ecrits dans un dossier local `shared_data/`: ils sont dans le volume Docker `bigdata-project_shared_data`, monte dans les conteneurs sur `/shared_data`.

Lister le contenu du volume:

```powershell
docker compose run --rm recommendation ls -R /shared_data
```

Afficher le fichier de tests:

```powershell
docker compose run --rm recommendation cat /shared_data/recommandation/test_results.json
```

Ouvrir un shell dans un conteneur avec le volume monte:

```powershell
docker compose run --rm recommendation bash
```

Puis, dans le conteneur:

```bash
ls -R /shared_data
cat /shared_data/recommandation/test_results.json
```

Copier les resultats du volume Docker vers un dossier local:

```powershell
docker compose cp recommendation:/shared_data ./shared_data_export
```

## Sorties Attendues

Apres une execution complete, le volume `/shared_data` doit contenir:

```text
/shared_data/images/
/shared_data/images_metadata.json
/shared_data/images_features.json
/shared_data/images_labels.json
/shared_data/visualizations/
/shared_data/recommandation/users.json
/shared_data/recommandation/recommendations.json
/shared_data/recommandation/test_results.json
```

Les visualisations principales sont:

- `/shared_data/visualizations/viz_collection_stats.png`
- `/shared_data/visualizations/viz_couleurs.png`
- `/shared_data/visualizations/viz_tags.png`
- `/shared_data/recommandation/cluster_distribution.png`
- `/shared_data/recommandation/tendances_utilisateurs.png`

## Verifier Le Bon Fonctionnement

Verifier la configuration Compose:

```powershell
docker compose config
```

Verifier que les trois conteneurs se terminent sans erreur:

```powershell
docker compose up --build
```

Verifier les tests de recommandation:

```powershell
docker compose run --rm recommendation cat /shared_data/recommandation/test_results.json
```

Le champ attendu est:

```json
{
  "all_passed": true
}
```

## Depannage

Si la collecte echoue avec une erreur SPARQL `502`, `504` ou `TimeoutException`, cela vient generalement du endpoint public Wikidata, qui peut etre surcharge. Le script d'acquisition utilise une requete SPARQL de secours plus legere, mais le service externe peut rester temporairement indisponible.

Dans ce cas:

```powershell
docker compose down -v
docker compose up --build
```

Si Docker affiche un avertissement du type `Access is denied` sur `C:\Users\<user>\.docker\config.json`, cela concerne la configuration locale Docker. Tant que `docker compose config` et `docker compose up` fonctionnent, ce warning ne bloque pas le pipeline.

