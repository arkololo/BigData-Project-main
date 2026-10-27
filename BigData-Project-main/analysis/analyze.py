# Service d'analyse: extrait les features, labels, clusters et graphiques.

import json
import logging
import os
import random
from collections import Counter
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Optional, Tuple
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans as SKKMeans
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

INPUT_DIR = os.getenv("INPUT_DIR", "/shared_data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/shared_data")
CLUSTER_K = int(os.getenv("CLUSTER_K", "8"))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "50000000"))

# Applique a tous les appels Pillow, y compris dans les workers Spark.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

# Fichiers echanges via le volume Docker partage par tous les services.
IMAGES_DIR = Path(INPUT_DIR) / "images"
METADATA_PATH = Path(INPUT_DIR) / "images_metadata.json"
FEATURES_PATH = Path(OUTPUT_DIR) / "images_features.json"
LABELS_PATH = Path(OUTPUT_DIR) / "images_labels.json"
VIZ_DIR = Path(OUTPUT_DIR) / "visualizations"

# 36 couleurs nommees, plus riches que la version initiale du notebook.
COLOR_NAMES: Dict[str, Tuple[int, int, int]] = {
    "red":        (255,   0,   0),
    "dark_red":   (139,   0,   0),
    "coral":      (255, 127,  80),
    "orange":     (255, 165,   0),
    "yellow":     (255, 255,   0),
    "green":      (  0, 128,   0),
    "dark_green": (  0, 100,   0),
    "sage":       (143, 188, 143),
    "olive":      (128, 128,   0),
    "blue":       (  0,   0, 255),
    "navy":       (  0,   0, 128),
    "sky_blue":   (135, 206, 235),
    "steel_blue": ( 70, 130, 180),
    "indigo":     ( 75,   0, 130),
    "purple":     (128,   0, 128),
    "lilac":      (200, 162, 200),
    "lavender":   (230, 230, 250),
    "pink":       (255, 192, 203),
    "magenta":    (255,   0, 255),
    "cyan":       (  0, 255, 255),
    "turquoise":  ( 64, 224, 208),
    "brown":      (139,  69,  19),
    "terracotta": (226, 114,  91),
    "beige":      (245, 222, 179),
    "peach":      (255, 218, 185),
    "cream":      (255, 253, 208),
    "ivory":      (255, 255, 240),
    "ochre":      (204, 119,  34),
    "gold":       (218, 165,  32),
    "khaki":      (189, 183, 107),
    "burgundy":   (128,   0,  32),
    "black":      (  0,   0,   0),
    "dark_gray":  ( 64,  64,  64),
    "gray":       (128, 128, 128),
    "light_gray": (192, 192, 192),
    "white":      (255, 255, 255),
}

# Caracteristiques normalisees utilisees pour le clustering PySpark KMeans.
KMEANS_FEATURES = [
    "width_norm", "height_norm", "aspect_ratio_norm",
    "mean_r_norm", "mean_g_norm", "mean_b_norm",
    "brightness", "contrast", "saturation", "warmth_score",
]


# ---------------------------------------------------------------------------
# Fonctions utilitaires pures
# ---------------------------------------------------------------------------

def nearest_color_name(rgb: Tuple[int, int, int]) -> str:
    best, best_dist = "unknown", float("inf")
    for name, ref in COLOR_NAMES.items():
        d = (rgb[0] - ref[0]) ** 2 + (rgb[1] - ref[1]) ** 2 + (rgb[2] - ref[2]) ** 2
        if d < best_dist:
            best_dist, best = d, name
    return best


def get_orientation(width: Optional[int], height: Optional[int]) -> str:
    if not width or not height:
        return "unknown"
    r = width / float(height)
    if r > 1.05:
        return "landscape"
    if r < 0.95:
        return "portrait"
    return "square"


def get_size_category(width: Optional[int], height: Optional[int]) -> str:
    if not width or not height:
        return "unknown"
    m = max(width, height)
    if m < 500:
        return "thumbnail"
    if m <= 1500:
        return "medium"
    return "large"


def brightness_label(b: float) -> str:
    if b < 0.35:
        return "dark"
    if b < 0.65:
        return "medium"
    return "bright"


def saturation_label(s: float) -> str:
    if s < 0.25:
        return "muted"
    if s < 0.55:
        return "normal"
    return "vivid"


def _year_tag(inception) -> Optional[str]:
    if not inception:
        return None
    y = str(inception)[:4]
    if not y.isdigit():
        return None
    year = int(y)
    if year < 1400:
        return "medieval"
    if year < 1600:
        return "renaissance"
    if year < 1800:
        return "classical"
    if year < 1900:
        return "modern"
    return "contemporary"


def _color_to_mpl(name: str) -> Tuple[float, float, float]:
    # Retourne une couleur (r, g, b) compatible matplotlib dans [0, 1].
    if name in COLOR_NAMES:
        r, g, b = COLOR_NAMES[name]
        return (r / 255.0, g / 255.0, b / 255.0)
    return (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Analyse image: une seule ouverture, toutes les metriques pixel calculees ensemble
# ---------------------------------------------------------------------------

def _analyze_image(image_path: Path, n_colors: int = 5) -> Optional[Dict]:
    
    # Ouvre l'image une seule fois, la redimensionne en 150x150 comme le notebook,
    # puis calcule les metriques RGB, luminosite, contraste, saturation, chaleur
    # et couleurs dominantes via sklearn KMeans.
    # Retourne None si l'image est invalide, absente ou trop grande.
    
    try:
        with Image.open(image_path) as img:
            raw_w, raw_h = img.size
            if raw_w * raw_h > MAX_PIXELS:
                logging.warning("Skip (too large %dx%d): %s", raw_w, raw_h, image_path.name)
                return None
            fmt = (img.format or "unknown").upper()
            img_rgb = img.convert("RGB")
            # 150x150 correspond a la taille des vignettes du notebook.
            img_rgb.thumbnail((150, 150))
            arr = np.array(img_rgb, dtype=np.float32)
    except Exception as exc:
        logging.warning("Skip (%s): %s — %s", type(exc).__name__, image_path.name, exc)
        return None

    pixels = arr.reshape(-1, 3)
    mean_rgb = pixels.mean(axis=0)

    # Luminosite perceptuelle (0-1).
    brightness = float(
        (0.299 * mean_rgb[0] + 0.587 * mean_rgb[1] + 0.114 * mean_rgb[2]) / 255.0
    )

    # Contraste: ecart-type normalise des niveaux de gris (0-1).
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    contrast = float(np.clip(gray.std() / 128.0, 0.0, 1.0))

    # Saturation: calcul HSV-S vectorise (0-1), sans boucle Python par pixel.
    norm = arr / 255.0
    cmax = norm.max(axis=2)
    cmin = norm.min(axis=2)
    delta = cmax - cmin
    with np.errstate(divide="ignore", invalid="ignore"):
        sat_map = np.where(cmax > 0, delta / cmax, 0.0)
    saturation = float(sat_map.mean())

    # Chaleur: equilibre (R - B) ramene dans [0, 1]; 0.5 = neutre.
    warmth = float(np.clip((mean_rgb[0] - mean_rgb[2]) / 255.0 * 0.5 + 0.5, 0.0, 1.0))

    # Couleurs dominantes via sklearn KMeans, comme dans le notebook.
    colors_rgb: List[List[int]] = []
    color_names_list: List[str] = []
    color_percentages: List[Dict] = []

    if len(pixels) >= n_colors:
        try:
            km = SKKMeans(n_clusters=n_colors, n_init=3, random_state=42)
            km.fit(pixels)
            counts = Counter(km.labels_.tolist())
            total = len(pixels)
            for idx, cnt in counts.most_common():
                rgb = [int(round(c)) for c in km.cluster_centers_[idx]]
                name = nearest_color_name((rgb[0], rgb[1], rgb[2]))
                colors_rgb.append(rgb)
                color_names_list.append(name)
                color_percentages.append({
                    "color": name,
                    "rgb": rgb,
                    "percentage": round(cnt / total, 4),
                })
        except Exception as exc:
            logging.warning("KMeans failed for %s: %s", image_path.name, exc)

    return {
        "raw_w": raw_w,
        "raw_h": raw_h,
        "format": fmt,
        "mean_r": float(round(mean_rgb[0], 4)),
        "mean_g": float(round(mean_rgb[1], 4)),
        "mean_b": float(round(mean_rgb[2], 4)),
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
        "saturation": round(saturation, 6),
        "warmth_score": round(warmth, 6),
        "predominant_colors": colors_rgb,
        "color_names": color_names_list,
        "color_percentages": color_percentages,
    }


def _normalize_record(record: Dict) -> Dict:
    # Normalise les cles de metadonnees francaises/anglaises vers le schema attendu.
    if "filename" in record:
        return record

    mapped = dict(record)
    if "nom_fichier" in record:
        mapped["filename"] = record.get("nom_fichier")
    if "chemin" in record:
        # Normalise les separateurs Windows s'ils sont presents dans le JSON.
        mapped["path"] = str(record.get("chemin")).replace("\\", "/")
    if "taille_ko" in record:
        mapped["size_kb"] = record.get("taille_ko")
    if "largeur" in record:
        mapped["width"] = record.get("largeur")
    if "hauteur" in record:
        mapped["height"] = record.get("hauteur")
    if "createur" in record:
        mapped["creator"] = record.get("createur")
    if "mouvement" in record:
        mapped["movement"] = record.get("mouvement")

    return mapped


def extract_features(record: Dict) -> Optional[Dict]:
    record = _normalize_record(record)
    file_name = record.get("filename")
    if not file_name:
        return None

    raw_path = record.get("path") or str(IMAGES_DIR / file_name)
    image_path = Path(raw_path)
    if not image_path.is_absolute():
        image_path = Path(INPUT_DIR) / image_path
    if not image_path.exists():
        logging.warning("Skip (not found): %s", file_name)
        return None

    # Les metriques pixel sont regroupees ici pour eviter d'ouvrir chaque image
    # plusieurs fois sur les workers Spark.
    img_data = _analyze_image(image_path)
    if img_data is None:
        return None

    # Preferer les dimensions des metadonnees (pleine resolution) a celles des vignettes.
    width = int(record.get("width") or img_data["raw_w"])
    height = int(record.get("height") or img_data["raw_h"])
    img_format = (record.get("format") or img_data["format"] or "unknown").upper()
    size_kb = float(
        record.get("size_kb") or round(image_path.stat().st_size / 1024.0, 2)
    )

    width_norm = round(min(width / 4096.0, 1.0), 6)
    height_norm = round(min(height / 4096.0, 1.0), 6)
    raw_aspect = width / float(height) if height else 1.0
    aspect_ratio = round(raw_aspect, 6)
    # Ratio d'aspect normalise par tanh pour garder les features KMeans dans [0, 1].
    aspect_ratio_norm = round(float(np.tanh((raw_aspect - 1.0) / 2.0) * 0.5 + 0.5), 6)

    b = img_data["brightness"]
    s = img_data["saturation"]

    tags = ["painting"]
    if record.get("movement"):
        tags.append(str(record["movement"]).strip().lower())
    if record.get("creator"):
        tags.append(str(record["creator"]).strip().lower())
    era = _year_tag(record.get("inception"))
    if era:
        tags.append(era)

    return {
        "filename": file_name,
        "path": str(image_path),
        "format": img_format,
        "width": width,
        "height": height,
        "width_norm": width_norm,
        "height_norm": height_norm,
        "aspect_ratio": aspect_ratio,
        "aspect_ratio_norm": aspect_ratio_norm,
        "size_kb": round(size_kb, 2),
        "mean_r": img_data["mean_r"],
        "mean_g": img_data["mean_g"],
        "mean_b": img_data["mean_b"],
        "mean_r_norm": round(img_data["mean_r"] / 255.0, 6),
        "mean_g_norm": round(img_data["mean_g"] / 255.0, 6),
        "mean_b_norm": round(img_data["mean_b"] / 255.0, 6),
        "brightness": b,
        "contrast": img_data["contrast"],
        "saturation": s,
        "warmth_score": img_data["warmth_score"],
        "predominant_colors": img_data["predominant_colors"],
        "color_names": img_data["color_names"],
        "color_percentages": img_data["color_percentages"],
        "orientation": get_orientation(width, height),
        "size_category": get_size_category(width, height),
        "brightness_label": brightness_label(b),
        "saturation_label": saturation_label(s),
        "tags": sorted(set(tags)),
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def save_visualizations(features: List[Dict], color_freq: Optional[Dict[str, int]] = None, tag_freq: Optional[Dict[str, int]] = None) -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    def _save(name: str):
        plt.tight_layout()
        plt.savefig(VIZ_DIR / name, dpi=150, bbox_inches="tight")
        plt.close()

    # VIZ 1: statistiques de collection (orientation, taille, format).
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Statistiques de la collection d'images", fontsize=16, fontweight="bold", y=1.02)

    orient_counts = Counter(f.get("orientation", "unknown") for f in features)
    orient_map = {"landscape": "paysage", "portrait": "portrait", "square": "carré", "unknown": "inconnu"}
    orient_order = ["paysage", "portrait", "carré", "inconnu"]
    orient_vals = [orient_counts.get(k, 0) for k, v in orient_map.items() if v in orient_order]
    orient_vals = [orient_counts.get(key, 0) for key in ["landscape", "portrait", "square", "unknown"]]
    colors_orient = ["#4CAF50", "#2196F3", "#FF9800", "#9E9E9E"]

    ax = axes[0]
    bars = ax.bar(orient_order, orient_vals, color=colors_orient, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, orient_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val),
                ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("Orientation")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Images par orientation")

    size_counts = Counter(f.get("size_category", "unknown") for f in features)
    size_order = ["thumbnail", "medium", "large", "unknown"]
    size_labels = ["vignette", "moyenne", "grande", "inconnu"]
    size_vals = [size_counts.get(s, 0) for s in size_order]
    colors_size = ["#FFC107", "#03A9F4", "#E91E63", "#9E9E9E"]

    ax = axes[1]
    bars = ax.bar(size_labels, size_vals, color=colors_size, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, size_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val),
                ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("Catégorie de taille")
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Images par catégorie de taille")

    def normalize_format(fmt: Optional[str]) -> Optional[str]:
        if not fmt:
            return None
        fmt = str(fmt).strip().upper()
        if fmt in {"JPG", "JPEG"}:
            return "JPEG/JPG"
        return fmt

    normalized_formats = [normalize_format(f.get("format")) for f in features]
    format_counts = Counter(fmt for fmt in normalized_formats if fmt)
    fmt_labels = list(format_counts.keys())
    fmt_vals = list(format_counts.values())
    colors_fmt = plt.cm.Set3(np.linspace(0, 1, len(fmt_labels)))

    total_fmt = sum(fmt_vals) if fmt_vals else 1
    labels_with_pct = [
        f"{lbl} ({val / total_fmt * 100:.1f}%)" for lbl, val in zip(fmt_labels, fmt_vals)
    ]

    ax = axes[2]
    ax.pie(
        fmt_vals,
        labels=labels_with_pct,
        colors=colors_fmt,
        startangle=90,
        labeldistance=1.05,
        textprops={"fontsize": 10, "fontweight": "bold"},
        rotatelabels=True,
    )
    ax.set_title("Distribution des formats d'image", pad=14, y=1.22)

    _save("viz_collection_stats.png")

    # VIZ 2: palettes de couleurs et histogramme des couleurs dominantes.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Analyse des couleurs de la collection", fontsize=16, fontweight="bold", y=1.02)

    ax = axes[0]
    ax.set_title("Palettes des 15 premières images")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 15)
    ax.set_yticks([])
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"Couleur {i+1}" for i in range(5)])

    sample_keys = [f for f in features if f.get("predominant_colors")][:15]
    for row_idx, feat in enumerate(sample_keys):
        colors = feat.get("predominant_colors")
        if not colors:
            continue
        y = 14 - row_idx
        for col_idx, rgb in enumerate(colors[:5]):
            color_norm = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
            rect = plt.Rectangle((col_idx, y), 0.9, 0.8, facecolor=color_norm,
                                 edgecolor="black", linewidth=0.3)
            ax.add_patch(rect)
        short = feat["filename"][:25] + ("…" if len(feat["filename"]) > 25 else "")
        ax.text(-0.1, y + 0.4, short, ha="right", va="center", fontsize=7)

    ax.set_xlim(-6, 5)

    _color_freq = color_freq or {}
    top_colors = sorted(_color_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    c_labels = [c for c, _ in top_colors]
    c_counts = [n for _, n in top_colors]
    c_bar_colors = [_color_to_mpl(name) for name in c_labels]

    ax = axes[1]
    x_pos = np.arange(len(c_labels))
    bars = ax.bar(x_pos, c_counts, color=c_bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(c_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Fréquence (nombre d'occurrences)")
    ax.set_title("Fréquence des couleurs prédominantes")
    for bar, val in zip(bars, c_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, str(val), ha="center", va="bottom", fontsize=9)

    _save("viz_couleurs.png")

    # VIZ 4: analyse des tags (barres et nuage de mots).
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Analyse des tags", fontsize=16, fontweight="bold", y=1.02)

    _tag_freq = tag_freq or {}
    top_tags_15 = sorted(_tag_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    t_labels = [t for t, _ in top_tags_15]
    t_counts = [n for _, n in top_tags_15]

    ax = axes[0]
    x_pos = np.arange(len(t_labels))
    colors_tags = plt.cm.viridis(np.linspace(0.2, 0.9, len(t_labels)))
    bars = ax.bar(x_pos, t_counts, color=colors_tags, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(t_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Tags les plus courants (top 15)")
    for bar, val in zip(bars, t_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, str(val), ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    ax.axis("off")
    ax.set_title("Nuage de tags")

    tag_freq_all = dict(sorted(_tag_freq.items(), key=lambda x: x[1], reverse=True)[:40])

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        colors = [
            "hsl(200, 80%%, %d%%)" % random.randint(30, 55),
            "hsl(260, 60%%, %d%%)" % random.randint(35, 55),
            "hsl(180, 70%%, %d%%)" % random.randint(30, 50),
            "hsl(220, 70%%, %d%%)" % random.randint(35, 55),
            "hsl(290, 50%%, %d%%)" % random.randint(40, 55),
        ]
        return random.choice(colors)

    wc = WordCloud(
        width=800,
        height=500,
        background_color="white",
        max_words=40,
        max_font_size=120,
        min_font_size=14,
        prefer_horizontal=1.0,
        relative_scaling=0.5,
        color_func=color_func,
        collocations=False,
        margin=4,
        random_state=42,
    )
    wc.generate_from_frequencies(tag_freq_all)

    ax.imshow(wc, interpolation="bilinear")

    _save("viz_tags.png")

    logging.info("Visualizations written to %s", VIZ_DIR)


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def main() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_records = json.load(f)

    read_count = len(metadata_records)
    logging.info("Images found in metadata: %s", read_count)

    spark = (
        SparkSession.builder.appName("image-analysis")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # Distribution des metadonnees: chaque worker ouvre une image et extrait
    # les caracteristiques visuelles utilisees pour les labels et le clustering.
    features_rdd = (
        sc.parallelize(metadata_records)
        .map(lambda rec: extract_features(rec))
        .filter(lambda rec: rec is not None)
        .cache()  # Evite de rouvrir les images entre count() et collect().
    )

    processed_count = features_rdd.count()
    logging.info(
        "Images processed: %s / %s — skipped: %s",
        processed_count, read_count, read_count - processed_count,
    )

    if processed_count == 0:
        raise RuntimeError("No valid image to analyze")

    features = features_rdd.collect()

    # Construire le DataFrame Spark pour KMeans: toutes les colonnes sont dans [0, 1].
    spark_rows = [
        {"filename": row["filename"], **{col: float(row[col]) for col in KMEANS_FEATURES}}
        for row in features
    ]
    feature_df = spark.createDataFrame(spark_rows)
    vector_df = VectorAssembler(inputCols=KMEANS_FEATURES, outputCol="features").transform(feature_df)

    if processed_count >= 2 and CLUSTER_K >= 2:
        k = min(CLUSTER_K, processed_count)
        model = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster").fit(vector_df)
        cluster_map = {
            row["filename"]: int(row["cluster"])
            for row in model.transform(vector_df).select("filename", "cluster").collect()
        }
        logging.info("KMeans clustering: %s clusters over %s images", k, processed_count)
    else:
        cluster_map = {row["filename"]: 0 for row in features}

    features_with_cluster: List[Dict] = []
    labels: List[Dict] = []

    # On garde un fichier de features riche pour les recommandations et un
    # fichier de labels plus leger pour les preferences utilisateur.
    for row in features:
        cluster = cluster_map.get(row["filename"], 0)
        features_with_cluster.append({**row, "cluster": cluster})
        labels.append({
            "filename": row["filename"],
            "color_names": row["color_names"],
            "dominant_color": row["color_names"][0] if row["color_names"] else None,
            "orientation": row["orientation"],
            "size_category": row["size_category"],
            "brightness_label": row["brightness_label"],
            "saturation_label": row["saturation_label"],
            "tags": row["tags"],
            "cluster": cluster,
        })

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(features_with_cluster, f, ensure_ascii=False, indent=2)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # Agregation des frequences avec Spark; les resultats alimentent les graphiques PNG.
    color_freq_rdd = (
        features_rdd
        .flatMap(lambda f: f.get("color_names", []))
        .map(lambda c: (c, 1))
        .reduceByKey(lambda a, b: a + b)
    )
    tag_freq_rdd = (
        features_rdd
        .flatMap(lambda f: f.get("tags", []))
        .map(lambda t: (t, 1))
        .reduceByKey(lambda a, b: a + b)
    )
    color_freq = dict(color_freq_rdd.collect())
    tag_freq = dict(tag_freq_rdd.collect())

    save_visualizations(features_with_cluster, color_freq=color_freq, tag_freq=tag_freq)

    logging.info("Features written to %s (%s entries)", FEATURES_PATH, len(features_with_cluster))
    logging.info("Labels written to %s (%s entries)", LABELS_PATH, len(labels))

    spark.stop()


if __name__ == "__main__":
    main()
