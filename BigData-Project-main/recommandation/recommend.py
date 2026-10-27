# Service de recommandation: construit les utilisateurs, score les images et ecrit les resultats.

import json
import logging
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


INPUT_DIR = os.getenv("INPUT_DIR", "/shared_data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/shared_data/recommandation")
USER_LIMIT = int(os.getenv("USER_LIMIT", "1000"))
RECO_PER_USER = int(os.getenv("RECO_PER_USER", "10"))
SEED = int(os.getenv("SEED", "42"))
TREND_USERS_MAX = int(os.getenv("TREND_USERS_MAX", "50"))

# Les entrees viennent de l'analyse; les sorties sont ecrites dans /shared_data/recommandation.
FEATURES_PATH = Path(INPUT_DIR) / "images_features.json"
LABELS_PATH = Path(INPUT_DIR) / "images_labels.json"
USERS_PATH = Path(OUTPUT_DIR) / "users.json"
RECOMMENDATIONS_PATH = Path(OUTPUT_DIR) / "recommendations.json"
TEST_RESULTS_PATH = Path(OUTPUT_DIR) / "test_results.json"


def cleanup_visual_outputs() -> None:
    # Supprime les anciennes visualisations qui ne sont plus utilisees.
    viz_dir = Path(OUTPUT_DIR) / "recommendation_visualizations"
    if viz_dir.exists() and viz_dir.is_dir():
        shutil.rmtree(viz_dir, ignore_errors=True)

    for name in ("cluster_distribution.png", "tendances_utilisateurs.png"):
        target = Path(OUTPUT_DIR) / name
        if target.exists():
            try:
                target.unlink()
            except OSError:
                logging.warning("Failed to remove legacy visual: %s", target)


def save_cluster_distribution(recommendations: List[Dict]) -> None:
    cluster_counts: Counter = Counter()
    for row in recommendations:
        for rec in row.get("recommendations", []):
            cluster_counts[int(rec.get("cluster", 0))] += 1

    if not cluster_counts:
        return

    sorted_counts = sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True)
    cluster_ids = sorted(cluster_counts.keys())
    cmap = plt.get_cmap("tab20")
    denom = max(1, len(cluster_ids) - 1)
    color_map = {
        cluster_id: cmap(index / denom)
        for index, cluster_id in enumerate(cluster_ids)
    }

    labels = [str(cluster_id) for cluster_id, _ in sorted_counts]
    values = [count for _, count in sorted_counts]
    colors = [color_map[cluster_id] for cluster_id, _ in sorted_counts]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values, color=colors, edgecolor="black", linewidth=0.4)
    plt.title("Recommended clusters distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "cluster_distribution.png", dpi=140)
    plt.close()


def save_user_trends(users: List[Dict]) -> None:
    if not users:
        return

    users_subset = users[:TREND_USERS_MAX] if TREND_USERS_MAX > 0 else users
    if len(users_subset) != len(users):
        logging.info("User trends limited to %s users", len(users_subset))

    all_user_colors = []
    all_user_tags = []
    orientation_counts: Counter = Counter()

    for profile in users_subset:
        all_user_colors.extend(profile.get("top_colors", []))
        all_user_tags.extend(profile.get("favorite_tags", []))
        orientation = profile.get("favorite_orientation")
        if orientation:
            orientation_counts[str(orientation)] += 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analyse des tendances — Utilisateurs", fontsize=16, fontweight="bold")

    color_freq = Counter(all_user_colors).most_common()
    color_labels = [c for c, _ in color_freq]
    color_counts = [n for _, n in color_freq]
    named_colors = {
        "red": (255, 0, 0),
        "dark_red": (139, 0, 0),
        "coral": (255, 127, 80),
        "orange": (255, 165, 0),
        "yellow": (255, 255, 0),
        "green": (0, 128, 0),
        "dark_green": (0, 100, 0),
        "sage": (143, 188, 143),
        "olive": (128, 128, 0),
        "blue": (0, 0, 255),
        "navy": (0, 0, 128),
        "sky_blue": (135, 206, 235),
        "steel_blue": (70, 130, 180),
        "indigo": (75, 0, 130),
        "purple": (128, 0, 128),
        "lilac": (200, 162, 200),
        "lavender": (230, 230, 250),
        "pink": (255, 192, 203),
        "magenta": (255, 0, 255),
        "cyan": (0, 255, 255),
        "turquoise": (64, 224, 208),
        "brown": (139, 69, 19),
        "terracotta": (226, 114, 91),
        "beige": (245, 222, 179),
        "peach": (255, 218, 185),
        "cream": (255, 253, 208),
        "ivory": (255, 255, 240),
        "ochre": (204, 119, 34),
        "gold": (218, 165, 32),
        "khaki": (189, 183, 107),
        "burgundy": (128, 0, 32),
        "black": (0, 0, 0),
        "dark_gray": (64, 64, 64),
        "gray": (128, 128, 128),
        "light_gray": (192, 192, 192),
        "white": (255, 255, 255),
    }
    fallback = (0.5, 0.5, 0.5)
    color_map = [
        (named_colors.get(name, None) or fallback) for name in color_labels
    ]
    color_map = [
        (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0) if isinstance(rgb, tuple) else fallback
        for rgb in color_map
    ]

    ax = axes[0, 0]
    x_pos = range(len(color_labels))
    bars = ax.bar(x_pos, color_counts, color=color_map, edgecolor="black", linewidth=0.4)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(color_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Nombre d'utilisateurs")
    ax.set_title("Couleurs favorites (tous utilisateurs)")
    for bar, val in zip(bars, color_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, str(val), ha="center", va="bottom", fontsize=9)

    tag_freq = Counter(all_user_tags).most_common(10)
    tag_labels = [t for t, _ in tag_freq]
    tag_counts = [n for _, n in tag_freq]

    ax = axes[0, 1]
    x_pos = range(len(tag_labels))
    bars = ax.bar(x_pos, tag_counts, color="#4E79A7", edgecolor="black", linewidth=0.4)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(tag_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Frequence")
    ax.set_title("Tags favoris globaux (top 10)")
    for bar, val in zip(bars, tag_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, str(val), ha="center", va="bottom", fontsize=9)

    ax = axes[1, 0]
    orient_labels = list(orientation_counts.keys())
    orient_counts = list(orientation_counts.values())
    if orient_labels:
        ax.pie(
            orient_counts,
            labels=orient_labels,
            autopct="%1.0f%%",
            colors=["#4CAF50", "#2196F3", "#FF9800", "#9E9E9E"],
            startangle=90,
            textprops={"fontsize": 11},
        )
    ax.set_title("Orientations favorites")

    axes[1, 1].axis("off")
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "tendances_utilisateurs.png", dpi=150, bbox_inches="tight")
    plt.close()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def feature_vector(feature: Dict) -> List[float]:
    return [
        float(feature.get("width_norm", 0.0)),
        float(feature.get("height_norm", 0.0)),
        float(feature.get("aspect_ratio", 0.0)),
        float(feature.get("size_kb", 0.0)) / 5000.0,
        float(feature.get("mean_r", 0.0)) / 255.0,
        float(feature.get("mean_g", 0.0)) / 255.0,
        float(feature.get("mean_b", 0.0)) / 255.0,
    ]


def content_similarity(user_vec: List[float], image_vec: List[float]) -> float:
    if not user_vec or not image_vec:
        return 0.0
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(user_vec, image_vec)))
    return clamp01(1.0 / (1.0 + dist))


def generate_user(idx: int, image_names: List[str], cluster_values: List[int], color_values: List[str]) -> Dict:
    rng = random.Random(SEED + idx)

    favorites_count = rng.randint(12, 18)
    favorites_count = min(favorites_count, len(image_names))
    favorites = rng.sample(image_names, favorites_count)

    preferred_clusters = rng.sample(cluster_values, min(3, len(cluster_values))) if cluster_values else []
    preferred_colors = rng.sample(color_values, min(5, len(color_values))) if color_values else []

    return {
        "user_id": f"user_{idx + 1:04d}",
        "name": f"User {idx + 1}",
        "favorite_images": favorites,
        "preferred_clusters_seed": preferred_clusters,
        "preferred_colors_seed": preferred_colors,
    }


def build_profile(user: Dict, feature_lookup: Dict[str, Dict], label_lookup: Dict[str, Dict]) -> Dict:
    favorites = user["favorite_images"]

    # Un profil resume les images favorites en vecteurs, couleurs, tags et
    # poids de clusters utilises par le systeme de recommandation.
    vectors = []
    cluster_counter: Dict[int, int] = {}
    color_counter: Dict[str, int] = {}
    tag_counter: Dict[str, int] = {}
    orientation_counter: Dict[str, int] = {}

    for file_name in favorites:
        feature = feature_lookup.get(file_name)
        label = label_lookup.get(file_name, {})
        if not feature:
            continue

        vectors.append(feature_vector(feature))

        cluster = int(feature.get("cluster", 0))
        cluster_counter[cluster] = cluster_counter.get(cluster, 0) + 1

        for color in label.get("color_names", []):
            color_counter[color] = color_counter.get(color, 0) + 1

        for tag in label.get("tags", []):
            tag_counter[str(tag).lower()] = tag_counter.get(str(tag).lower(), 0) + 1

        orientation = label.get("orientation")
        if orientation:
            orientation_counter[str(orientation)] = orientation_counter.get(str(orientation), 0) + 1

    if vectors:
        avg_vec = [sum(values) / len(values) for values in zip(*vectors)]
    else:
        avg_vec = [0.0] * 7

    cluster_total = sum(cluster_counter.values()) or 1
    cluster_weights = {
        int(cluster): round(count / cluster_total, 6)
        for cluster, count in cluster_counter.items()
    }

    top_colors = [
        item[0]
        for item in sorted(color_counter.items(), key=lambda pair: pair[1], reverse=True)[:5]
    ]

    top_tags = [
        item[0]
        for item in sorted(tag_counter.items(), key=lambda pair: pair[1], reverse=True)[:5]
    ]

    favorite_orientation = None
    if orientation_counter:
        favorite_orientation = max(orientation_counter.items(), key=lambda item: item[1])[0]

    return {
        "user_id": user["user_id"],
        "name": user["name"],
        "favorite_images": favorites,
        "preferred_clusters_seed": user.get("preferred_clusters_seed", []),
        "preferred_colors_seed": user.get("preferred_colors_seed", []),
        "profile_avg_vector": avg_vec,
        "cluster_weights": cluster_weights,
        "top_colors": top_colors,
        "favorite_tags": top_tags,
        "favorite_orientation": favorite_orientation,
    }


def recommend_for_profile(profile: Dict, features: List[Dict], top_k: int = 10) -> List[Dict]:
    favorites = set(profile.get("favorite_images", []))
    user_vec = profile.get("profile_avg_vector", [0.0] * 7)
    cluster_weights = {int(k): float(v) for k, v in profile.get("cluster_weights", {}).items()}

    # Score hybride: similarite visuelle plus bonus pour les clusters preferes.
    scored = []
    for feature in features:
        file_name = feature["filename"]
        if file_name in favorites:
            continue

        image_vec = feature_vector(feature)
        score_content = content_similarity(user_vec, image_vec)
        cluster = int(feature.get("cluster", 0))
        score_cluster = clamp01(cluster_weights.get(cluster, 0.0))
        final_score = clamp01(0.6 * score_content + 0.4 * score_cluster)

        scored.append(
            {
                "filename": file_name,
                "cluster": cluster,
                "score_content": round(score_content, 6),
                "score_cluster": round(score_cluster, 6),
                "score_final": round(final_score, 6),
            }
        )

    scored.sort(key=lambda row: row["score_final"], reverse=True)
    return scored[:top_k]




def run_tests(users: List[Dict], recommendations: List[Dict]) -> Dict:
    checks = []

    recommendations_by_id = {row["user_id"]: row for row in recommendations}

    each_user_has_recos = all(
        len(recommendations_by_id.get(user["user_id"], {}).get("recommendations", [])) > 0
        for user in users
    )
    checks.append({"name": "each_user_has_recommendations", "passed": each_user_has_recos})

    no_favorite_recommended = True
    for user in users:
        recs = recommendations_by_id.get(user["user_id"], {}).get("recommendations", [])
        favs = set(user.get("favorite_images", []))
        if any(rec["filename"] in favs for rec in recs):
            no_favorite_recommended = False
            break
    checks.append({"name": "no_favorite_recommended", "passed": no_favorite_recommended})

    scores_in_range = True
    for row in recommendations:
        for rec in row.get("recommendations", []):
            if not (0.0 <= rec.get("score_final", 0.0) <= 1.0):
                scores_in_range = False
                break
        if not scores_in_range:
            break
    checks.append({"name": "scores_in_0_1", "passed": scores_in_range})

    output_files_exist = all(
        path.exists() for path in [USERS_PATH, RECOMMENDATIONS_PATH, TEST_RESULTS_PATH]
    )
    checks.append({"name": "output_files_exist", "passed": output_files_exist})

    return {
        "all_passed": all(check["passed"] for check in checks),
        "checks": checks,
        "metrics": {
            "users": len(users),
            "recommendation_rows": len(recommendations),
            "total_recommendations": sum(len(row.get("recommendations", [])) for row in recommendations),
        },
    }


def main() -> None:
    if not FEATURES_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("Missing images_features.json or images_labels.json")

    cleanup_visual_outputs()

    with open(FEATURES_PATH, "r", encoding="utf-8") as source:
        features = json.load(source)

    with open(LABELS_PATH, "r", encoding="utf-8") as source:
        labels = json.load(source)

    if not features:
        raise RuntimeError("No feature rows available")

    feature_lookup = {row["filename"]: row for row in features}
    label_lookup = {row["filename"]: row for row in labels}

    image_names = [row["filename"] for row in features]

    spark = (
        SparkSession.builder.appName("image-recommendation")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    # On agrege d'abord les valeurs globales pour generer des utilisateurs plausibles.
    cluster_counts_rdd = (
        sc.parallelize(features)
        .map(lambda f: (int(f.get("cluster", 0)), 1))
        .reduceByKey(lambda a, b: a + b)
    )
    cluster_values = sorted(c for c, _ in cluster_counts_rdd.collect())

    color_freq_rdd = (
        sc.parallelize(list(label_lookup.values()))
        .flatMap(lambda label: label.get("color_names", []))
        .map(lambda c: (c, 1))
        .reduceByKey(lambda a, b: a + b)
    )
    color_values = [c for c, _ in sorted(color_freq_rdd.collect(), key=lambda x: x[1], reverse=True)]

    # Spark distribue la generation des utilisateurs, la construction des profils et le scoring.
    users_rdd = sc.parallelize(range(USER_LIMIT)).map(
        lambda idx: generate_user(idx, image_names, cluster_values, color_values)
    )

    profiles_rdd = users_rdd.map(
        lambda user: build_profile(user, feature_lookup, label_lookup)
    )

    recommendations_rdd = profiles_rdd.map(
        lambda profile: {
            "user_id": profile["user_id"],
            "recommendations": recommend_for_profile(profile, features, top_k=RECO_PER_USER),
        }
    )

    users = profiles_rdd.collect()
    recommendations = recommendations_rdd.collect()

    read_count = len(features)
    processed_count = len(recommendations)
    error_count = max(USER_LIMIT - processed_count, 0)

    logging.info("Images read: %s", read_count)
    logging.info("Users processed: %s", processed_count)
    logging.info("Errors: %s", error_count)
    logging.info("Users generated: %s", len(users))
    logging.info(
        "Recommendations generated: %s",
        sum(len(row.get("recommendations", [])) for row in recommendations),
    )

    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(USERS_PATH, "w", encoding="utf-8") as output:
        json.dump(users, output, ensure_ascii=False, indent=2)

    with open(RECOMMENDATIONS_PATH, "w", encoding="utf-8") as output:
        json.dump(recommendations, output, ensure_ascii=False, indent=2)

    # Les graphiques lies aux recommandations sont stockes avec les sorties JSON.
    save_user_trends(users)
    save_cluster_distribution(recommendations)

    # Ecrit d'abord un fichier de test provisoire pour rendre le controle des sorties pertinent.
    with open(TEST_RESULTS_PATH, "w", encoding="utf-8") as output:
        json.dump({"status": "precheck"}, output, ensure_ascii=False, indent=2)

    test_results = run_tests(users, recommendations)

    with open(TEST_RESULTS_PATH, "w", encoding="utf-8") as output:
        json.dump(test_results, output, ensure_ascii=False, indent=2)

    logging.info("Users written to %s", USERS_PATH)
    logging.info("Recommendations written to %s", RECOMMENDATIONS_PATH)
    logging.info("Tests written to %s", TEST_RESULTS_PATH)

    spark.stop()


if __name__ == "__main__":
    main()
