# Service d'acquisition: recupere les URLs, telecharge les images et ecrit les metadonnees.

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS
from requests.adapters import HTTPAdapter
from SPARQLWrapper import JSON, POST, POSTDIRECTLY, SPARQLWrapper
from urllib3.util.retry import Retry


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/shared_data")
IMAGE_DIR = Path(OUTPUT_DIR) / "images"
METADATA_PATH = Path(OUTPUT_DIR) / "images_metadata.json"
IMAGE_LIMIT = int(os.getenv("IMAGE_LIMIT", "1000"))
# On recupere un peu plus de candidats pour compenser les echecs de telechargement.
CANDIDATE_FACTOR = float(os.getenv("CANDIDATE_FACTOR", "1.5"))
# Wikimedia peut limiter les requetes, donc le nombre de workers reste configurable.
MAX_WORKERS = int(os.getenv("DOWNLOAD_WORKERS", "6"))

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
# User-Agent descriptif requis par la politique Wikimedia.
USER_AGENT = "BigDataProject-CPE/1.0 (educational project; contact: student@cpe.fr)"


def run_query(query: str, retries: int = 5, base_wait: int = 2) -> Dict:
    sparql = SPARQLWrapper(SPARQL_ENDPOINT, agent=USER_AGENT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setMethod(POST)
    sparql.setRequestMethod(POSTDIRECTLY)
    sparql.setTimeout(120)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return sparql.query().convert()
        except Exception as exc:
            last_error = exc
            wait = base_wait * attempt
            logging.warning("SPARQL attempt %s/%s failed: %s", attempt, retries, exc)
            time.sleep(wait)

    raise RuntimeError(f"SPARQL failed after {retries} attempts: {last_error}")


def chunked(items: List, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_candidates(limit: int) -> List[Dict]:
    # On tente d'abord la requete legere; la requete par sous-classes est plus
    # complete, mais elle peut expirer quand le endpoint public Wikidata est charge.
    queries = [
        (
            "direct painting instances",
            f"""
    SELECT ?item ?image ?sitelinks
    WHERE {{
      ?item wdt:P31 wd:Q3305213 .
      ?item wdt:P18 ?image .
      ?item wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks > 10)
    }}
    LIMIT {limit}
    """,
        ),
        (
            "subclass painting instances ordered by sitelinks",
            f"""
    SELECT ?item ?image ?sitelinks
    WHERE {{
      ?item wdt:P31/wdt:P279* wd:Q3305213 .
      ?item wdt:P18 ?image .
      ?item wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks > 10)
    }}
    ORDER BY DESC(?sitelinks)
    LIMIT {limit}
    """,
        ),
    ]

    results = None
    last_error = None
    for name, query in queries:
        try:
            logging.info("Fetching candidates with SPARQL strategy: %s", name)
            results = run_query(query)
            break
        except Exception as exc:
            last_error = exc
            logging.warning("Candidate query strategy failed (%s): %s", name, exc)

    if results is None:
        raise RuntimeError(f"All candidate SPARQL strategies failed: {last_error}")

    rows = results.get("results", {}).get("bindings", [])

    seen = set()
    candidates: List[Dict] = []
    for row in rows:
        item = row.get("item", {}).get("value")
        image = row.get("image", {}).get("value")
        sitelinks_raw = row.get("sitelinks", {}).get("value", "0")
        if not item or not image or item in seen:
            continue
        seen.add(item)
        candidates.append({"item": item, "image": image, "sitelinks": int(sitelinks_raw)})

    return candidates


def fetch_item_metadata(items: List[str], batch_size: int = 50) -> Dict[str, Dict]:
    metadata_map: Dict[str, Dict] = {}

    for i, batch in enumerate(chunked(items, batch_size)):
        # Les metadonnees sont recuperees par petits lots pour eviter de trop grosses requetes SPARQL.
        values = " ".join(f"<{uri}>" for uri in batch)
        query = f"""SELECT ?item
  (SAMPLE(?itemLabel_) AS ?itemLabel)
  (SAMPLE(?creatorLabel_) AS ?creatorLabel)
  (SAMPLE(?inception_) AS ?inception)
  (SAMPLE(?movementLabel_) AS ?movementLabel)
  (SAMPLE(?licenseLabel_) AS ?licenseLabel)
WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item wdt:P170 ?creator_. ?creator_ rdfs:label ?creatorLabel_ . FILTER(LANG(?creatorLabel_) IN ("fr","en")) }}
  OPTIONAL {{ ?item wdt:P571 ?inception_. }}
  OPTIONAL {{ ?item wdt:P135 ?movement_. ?movement_ rdfs:label ?movementLabel_ . FILTER(LANG(?movementLabel_) IN ("fr","en")) }}
  OPTIONAL {{ ?item wdt:P275 ?license_. ?license_ rdfs:label ?licenseLabel_ . FILTER(LANG(?licenseLabel_) IN ("fr","en")) }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "fr,en". ?item rdfs:label ?itemLabel_. }}
}}
GROUP BY ?item"""

        results = run_query(query)
        rows = results.get("results", {}).get("bindings", [])
        for row in rows:
            uri = row.get("item", {}).get("value")
            if not uri:
                continue
            metadata_map[uri] = {
                "title": row.get("itemLabel", {}).get("value"),
                "creator": row.get("creatorLabel", {}).get("value"),
                "inception": row.get("inception", {}).get("value"),
                "movement": row.get("movementLabel", {}).get("value"),
                "license": row.get("licenseLabel", {}).get("value"),
            }

        logging.info("Metadata batch %s done: %s rows", i + 1, len(rows))
        time.sleep(1)

    return metadata_map


def sanitize_filename(url: str) -> str:
    parsed = urlparse(url)
    name = unquote(os.path.basename(parsed.path))
    name = re.sub(r'[<>:"/\\|?*]', "_", name).strip()
    if not name:
        name = "image_unknown.jpg"
    if len(name) > 220:
        stem, ext = os.path.splitext(name)
        name = f"{stem[:210]}{ext}"
    return name


def _make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _fetch_to_disk(image_url: str, file_path: Path, session: requests.Session) -> bool:
    # Télécharge une URL vers file_path via un fichier .part temporaire. Retourne True si ok.
    temp_path = file_path.with_suffix(file_path.suffix + ".part")
    try:
        response = session.get(image_url, stream=True, timeout=(15, 90), allow_redirects=True)
        if response.status_code != 200:
            logging.warning("Skip %s: HTTP %s", image_url, response.status_code)
            return False

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type and "octet-stream" not in content_type:
            logging.warning("Skip %s: type inattendu %s", image_url, content_type)
            return False

        response.raw.decode_content = True
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

        if temp_path.stat().st_size < 1024:
            logging.warning("Skip %s: fichier trop petit", image_url)
            temp_path.unlink(missing_ok=True)
            return False

        temp_path.replace(file_path)
        return True
    except Exception as exc:
        logging.warning("Echec téléchargement %s: %s", image_url, exc)
        temp_path.unlink(missing_ok=True)
        return False


def _read_image_info(file_path: Path) -> Dict:
    # Lit dimensions, format et EXIF d'une image. Retourne des None en cas d'échec.
    try:
        with Image.open(file_path) as img:
            exif_clean = {}
            for tag_id, value in list(img.getexif().items())[:30]:
                tag_name = TAGS.get(tag_id, str(tag_id))
                try:
                    json.dumps(value)
                    exif_clean[tag_name] = value
                except (TypeError, ValueError):
                    exif_clean[tag_name] = str(value)
            return {"width": img.width, "height": img.height, "format": img.format, "exif": exif_clean or None}
    except UnidentifiedImageError:
        raise
    except Exception as exc:
        logging.warning("Lecture image échouée %s: %s", file_path.name, exc)
        return {"width": None, "height": None, "format": None, "exif": None}


def download_one(record: Dict, session: requests.Session) -> Optional[Dict]:
    image_url = record["image"]
    file_name = sanitize_filename(image_url)
    file_path = IMAGE_DIR / file_name

    if not file_path.exists() and not _fetch_to_disk(image_url, file_path, session):
        return None

    try:
        info = _read_image_info(file_path)
    except UnidentifiedImageError:
        logging.warning("Image invalide, supprimée: %s", file_name)
        file_path.unlink(missing_ok=True)
        return None

    return {
        "filename": file_name,
        "path": str(file_path),
        "size_kb": round(file_path.stat().st_size / 1024.0, 2),
        "source_url": image_url,
        "item": record.get("item"),
        "sitelinks": record.get("sitelinks"),
        "title": record.get("title"),
        "creator": record.get("creator"),
        "inception": record.get("inception"),
        "movement": record.get("movement"),
        "license": record.get("license") or "Public domain (Wikimedia Commons)",
        "width": info["width"],
        "height": info["height"],
        "format": info["format"],
        "exif": info["exif"],
    }


def _download_all(enriched: List[Dict], session: requests.Session, limit: int):
    # Télécharge par batches de 50 avec MAX_WORKERS threads, s'arrête dès que limit est atteint.
    downloaded: List[Dict] = []
    failed = 0

    for batch_num, batch in enumerate(chunked(enriched, 50)):
        if len(downloaded) >= limit:
            break

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(download_one, record, session): record["image"] for record in batch}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    logging.warning("Erreur worker pour %s: %s", futures[future], exc)
                    failed += 1
                    continue
                if result is None:
                    failed += 1
                else:
                    downloaded.append(result)

        logging.info("Batch %s — %s/%s téléchargées, %s échecs", batch_num + 1, len(downloaded), limit, failed)

    return downloaded[:limit], failed


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if IMAGE_LIMIT <= 0:
        raise ValueError("IMAGE_LIMIT must be > 0")

    candidate_limit = max(IMAGE_LIMIT, int(IMAGE_LIMIT * CANDIDATE_FACTOR))
    logging.info("IMAGE_LIMIT=%s candidate_limit=%s workers=%s", IMAGE_LIMIT, candidate_limit, MAX_WORKERS)

    # 1) Recuperer les items Wikidata candidats et les enrichir avec des metadonnees lisibles.
    candidates = fetch_candidates(candidate_limit)
    logging.info("Candidats récupérés via SPARQL: %s", len(candidates))
    if not candidates:
        raise RuntimeError("Aucun candidat trouvé via SPARQL")

    items = [r["item"] for r in candidates]
    meta_map = fetch_item_metadata(items)

    enriched: List[Dict] = []
    for row in candidates:
        enriched.append({**row, **meta_map.get(row["item"], {})})

    # 2) Telecharger les images, les valider avec Pillow, puis ecrire le JSON partage.
    session = _make_session()
    downloaded, failed = _download_all(enriched, session, IMAGE_LIMIT)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(downloaded, f, ensure_ascii=False, indent=2, default=str)

    logging.info("Images téléchargées: %s", len(downloaded))
    logging.info("Echecs: %s", failed)
    logging.info("Métadonnées écrites dans: %s", METADATA_PATH)


if __name__ == "__main__":
    main()
