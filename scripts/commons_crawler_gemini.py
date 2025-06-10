#!/usr/bin/env python3
"""commons_crawler.py — download winged-insect images from Wikimedia Commons

* Refactored to prioritize category-based crawling for better coverage.
* Uses the MediaWiki API more extensively for category traversal.
* Improved error handling and logging.
* Throttles SDC queries (1s) and paginates (LIMIT 500), MediaWiki API (0.1s).
* FIX: Robustified category member processing in get_files_from_category
       to handle cases where 'categorymembers' might be missing or empty,
       or individual member dictionaries lack expected keys (e.g., 'type').
* FIX: Switched SPARQL query execution from SPARQLWrapper to requests for
       more robust handling of redirects (e.g., HTTP 307) and network errors.
* FIX: Ensured SPARQL queries explicitly request JSON format using 'Accept' header
       to prevent 'JSONDecodeError' from XML responses.
* NEW: Improved filtering in get_files_from_category based on namespace (ns=6 for files, ns=14 for categories).
* TEMP: Temporarily disabled SDC (P180) queries due to authentication redirects.
* NEW: Implemented periodic saving of the manifest.csv to mitigate data loss on long runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from random import uniform
from pathlib import Path
import urllib.parse
from typing import Dict, List, Set, Tuple, Any

import pandas as pd
import requests

# --- Configuration ---
WD_ENDPOINT = "https://query.wikidata.org/sparql"
SDC_ENDPOINT = "https://commons-query.wikimedia.org/sparql"
MW_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
# IMPORTANT: Change this User-Agent to your actual contact info or project URL!
HEADERS = {
    "User-Agent": "WingImageCrawler/1.5 (contact: your.email@example.com)",
    "Accept": "application/sparql-results+json",  # Crucial for JSON SPARQL responses
}
logger = logging.getLogger("commons_crawler")


# --- SPARQL Helper (POST + retry with back-off) ---
def run_sparql(endpoint: str, query: str, max_tries: int = 6) -> dict | None:
    """
    Executes a SPARQL query using requests with retries and exponential back-off.
    Handles common HTTP errors and JSON decoding issues.
    """
    params = {"query": query}
    for attempt in range(1, max_tries + 1):
        try:
            # Use POST for SPARQL queries as they can be long
            # Allow redirects by default for WD, but for SDC we might need to be careful
            # Given the previous SDC issue, let's keep allow_redirects=True for now,
            # but note it's leading to login pages for SDC.
            r = requests.post(
                endpoint, data=params, headers=HEADERS, timeout=60, allow_redirects=True
            )
            r.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            try:
                res = r.json()
            except json.JSONDecodeError as e:
                logger.error(
                    "JSON decoding error for SPARQL response from %s: %s", endpoint, e
                )
                logger.error(
                    "Raw response content (first 500 chars): %s", r.text[:500]
                )  # Print first 500 chars of raw content
                raise  # Re-raise to be caught by the outer except block

            return res
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            wait = 2 ** (attempt - 1) + uniform(0, 1)
            logger.warning(
                "SPARQL query failed (attempt %d/%d) for endpoint %s. Retrying in %.1fs. Error: %s",
                attempt,
                max_tries,
                endpoint,
                wait,
                e,
            )
            time.sleep(wait)
        except Exception as e:  # Catch any other unexpected errors
            logger.error(
                "Unexpected error during SPARQL query (attempt %d/%d): %s",
                attempt,
                max_tries,
                e,
            )
            return None  # Fail immediately for unexpected errors
    logger.error(
        "SPARQL query failed after %d attempts for endpoint: %s", max_tries, endpoint
    )
    return None


# --- Step 1 – Get initial taxa and associated categories from Wikidata ---
def get_initial_taxa_and_categories(clade_qid: str) -> Dict[str, List[str]]:
    """
    Retrieves descendant taxa from a given clade that have images and their
    associated Commons categories from Wikidata (P373).
    Crucially, it now explicitly filters to ensure only 'Insecta' taxa are included.
    """
    logger.info(
        "Fetching descendant taxa and associated categories from Wikidata for clade %s, ensuring they are Insecta...",
        clade_qid,
    )
    q = f"""
    SELECT DISTINCT ?taxon ?commonsCategory WHERE {{
      ?taxon wdt:P171* wd:{clade_qid} . # Descendants of the specified clade
      ?taxon wdt:P31/wdt:P279* wd:Q13689 . # AND: The taxon must be an instance of or subclass of Insecta (Q13689)
      OPTIONAL {{ ?taxon wdt:P373 ?commonsCategory . }}
      FILTER EXISTS {{ ?taxon wdt:P18 ?img . }} # Only include taxa with at least one image
    }}
    """
    data = run_sparql(WD_ENDPOINT, q)
    if not data:
        logger.error("Failed to retrieve descendant taxa from Wikidata.")
        return {}

    taxa_categories: Dict[str, List[str]] = {}
    for b in data.get("results", {}).get("bindings", []):
        taxon_qid = b.get("taxon", {}).get("value", "").split("/")[-1]
        category = b.get("commonsCategory", {}).get("value")
        if taxon_qid and taxon_qid not in taxa_categories:
            taxa_categories[taxon_qid] = []
        if taxon_qid and category:
            taxa_categories[taxon_qid].append(category)

    logger.info("Found %d descendant taxa (filtered for Insecta).", len(taxa_categories))
    return taxa_categories


# --- Step 2 – Commons File Retrieval (Category-based and P180/P18) ---


def get_files_from_category(
    category_name: str, max_depth: int = 2, max_files_per_category: int = 5000
) -> Set[str]:
    """
    Recursively fetches file titles from a given Wikimedia Commons category
    and its subcategories up to a specified depth.
    """
    logger.info(
        "Starting category crawl for '%s' (max_depth: %d)...", category_name, max_depth
    )
    all_titles: Set[str] = set()
    categories_to_process: List[Tuple[str, int]] = [(category_name, 0)]
    processed_categories: Set[str] = set()

    while categories_to_process:
        current_cat_name, current_depth = categories_to_process.pop(0)

        if current_cat_name in processed_categories:
            continue
        processed_categories.add(current_cat_name)

        if current_depth > max_depth:
            continue

        logger.debug(
            "Processing category: '%s' (Depth: %d)", current_cat_name, current_depth
        )

        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Category:{current_cat_name}",
            "cmtype": "file|subcat",  # Get both files and subcategories
            "cmlimit": "500",  # Max limit per request
            "formatversion": "2",
        }
        cmcontinue = None
        current_cat_files_count = 0

        while True:
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            try:
                r = requests.get(
                    MW_ENDPOINT, params=params, headers=HEADERS, timeout=30
                )
                r.raise_for_status()
                data = r.json()

                members = data.get("query", {}).get("categorymembers")
                if not isinstance(members, list):
                    logger.debug(
                        "No category members found or unexpected data for category '%s'. Data: %s",
                        current_cat_name,
                        data,
                    )
                    break  # Exit inner while loop, no more members for this category

                for m in members:
                    member_title = m.get("title")
                    member_ns = m.get("ns")  # Get the namespace
                    member_type = m.get("type") # Get the type field

                    if not member_title:
                        logger.debug(
                            "Skipping category member due to missing 'title': %s", m
                        )
                        continue

                    # Handle cases where 'type' might be None, infer from namespace (ns)
                    # Namespace 6 is for files, Namespace 14 is for categories
                    if member_ns == 6: # It's a file
                        title = member_title.replace("File:", "")
                        if title not in all_titles:
                            all_titles.add(title)
                            current_cat_files_count += 1
                            if current_cat_files_count >= max_files_per_category:
                                logger.warning(
                                    "Reached max files (%d) for category '%s'. Stopping further file collection in this category.",
                                    max_files_per_category,
                                    current_cat_name,
                                )
                                break  # Stop processing files for this category
                    elif member_ns == 14: # It's a subcategory
                        subcat_name = member_title.replace("Category:", "")
                        categories_to_process.append(
                            (subcat_name, current_depth + 1)
                        )
                    else: # Neither file nor subcategory namespace
                        logger.debug(
                            "Skipping category member '%s' of unexpected type '%s' (ns=%s).",
                            member_title,
                            member_type, # This will now log 'None' if it's indeed None
                            member_ns,
                        )

                cmcontinue = data.get("continue", {}).get("cmcontinue")
                if not cmcontinue or current_cat_files_count >= max_files_per_category:
                    break
                time.sleep(0.1)  # Be polite to MediaWiki API
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "MediaWiki API error during category '%s' fetch: %s",
                    current_cat_name,
                    e,
                )
                break
            except Exception as e:
                logger.error(
                    "Unexpected error during category member processing for '%s': %s",
                    current_cat_name,
                    e,
                )
                logger.debug(
                    "Problematic member data: %s", m
                )  # Log the problematic member
                break  # Break to move to next category if an issue with data structure occurs

    logger.info(
        "Finished category crawl for '%s'. Found %d unique file titles.",
        category_name,
        len(all_titles),
    )
    return all_titles


def get_files_for_taxon_via_sdc(taxon_qid: str) -> Set[str]:
    """
    Retrieves file titles depicting *taxon_qid* using Commons SDC (P180).
    """
    logger.debug("Fetching files for taxon %s via SDC (P180)...", taxon_qid)
    titles: Set[str] = set()
    offset, page = 0, 500
    while True:
        q = (
            f"SELECT ?file WHERE {{ ?file wdt:P180 wd:{taxon_qid} . }} "
            f"LIMIT {page} OFFSET {offset}"
        )
        # Note: SDC endpoint is currently experiencing authentication issues.
        # This part is likely to fail until that's resolved or a workaround is found.
        data = run_sparql(SDC_ENDPOINT, q, max_tries=4)
        time.sleep(1.0)  # 1 rps throttle for SDC
        if not data:
            logger.debug(
                "No more SDC data or SDC query failed for %s (possible authentication issue).",
                taxon_qid,
            )
            break
        # Access results via .get() to be safe
        batch = {
            # Normalize filenames from SPARQL URLs to unquote then use underscores
            urllib.parse.unquote(b.get("file", {}).get("value", "").split("/")[-1]).replace(' ', '_')
            for b in data.get("results", {}).get("bindings", [])
            if b.get("file", {}).get("value")
        }  # Ensure value exists
        if not batch:
            break
        titles.update(batch)
        if len(batch) < page:
            break
        offset += page
    logger.debug("Found %d files for taxon %s via SDC.", len(titles), taxon_qid)
    return titles


def get_files_for_taxon_via_wikidata_p18(taxon_qid: str) -> Set[str]:
    """
    Retrieves the single representative image (P18) from a Wikidata taxon item.
    """
    logger.debug("Fetching P18 image for taxon %s from Wikidata...", taxon_qid)
    q_wd = f"SELECT ?file WHERE {{ wd:{taxon_qid} wdt:P18 ?file . }}"
    wd_data = run_sparql(WD_ENDPOINT, q_wd, max_tries=3)
    if wd_data:
        titles = {
            # Normalize filenames from SPARQL URLs to unquote then use underscores
            urllib.parse.unquote(b.get("file", {}).get("value", "").split("/")[-1]).replace(' ', '_')
            for b in wd_data.get("results", {}).get("bindings", [])
            if b.get("file", {}).get("value")
        }  # Ensure value exists
        logger.debug("Found %d P18 images for taxon %s.", len(titles), taxon_qid)
        return titles
    logger.debug("No P18 image found for taxon %s.", taxon_qid)
    return set()


# --- MediaWiki Helpers (ImageInfo and Download) ---


def fetch_imageinfo(titles: List[str]) -> Dict[str, dict]:
    """
    Fetches detailed image information from the MediaWiki API for a list of titles.
    """
    if not titles:
        return {}
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|sha1|size|extmetadata|mime",  # Added mime type
        # The titles coming into this function should now already be normalized
        # to use underscores and literal characters for MediaWiki API.
        "titles": "|".join([f"File:{t}" for t in titles]),
        "formatversion": "2",
    }
    try:
        r = requests.get(MW_ENDPOINT, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning(
            "MediaWiki API imageinfo error for %d titles: %s", len(titles), e
        )
        return {}
    info: Dict[str, dict] = {}
    for page in r.json().get("query", {}).get("pages", []):
        meta = page.get("imageinfo")
        # Ensure imageinfo exists and is not empty, and that 'title' is present
        if meta and meta[0] and page.get("title"):
            title = page.get("title").replace("File:", "")
            info[title] = meta[0]
        else:
            logger.debug(
                "No imageinfo found for file: %s (or page not found)",
                page.get("title", "UNKNOWN"),
            )
    return info


def download(url: str, dest: Path) -> None:
    """
    Downloads a file from a given URL to a destination path.
    Creates parent directories if they don't exist and skips if file already exists.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (
        dest.exists() and dest.stat().st_size > 0
    ):  # Check if file exists and is not empty
        logger.debug("File already exists, skipping download: %s", dest.name)
        return

    try:
        r = requests.get(
            url, headers=HEADERS, timeout=60, stream=True
        )  # Use stream=True for large files
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.debug("Downloaded: %s", dest.name)
    except requests.exceptions.RequestException as e:
        # Re-raise to be caught by the caller for specific logging
        raise ConnectionError(f"Download request failed for {url}: {e}") from e
    except OSError as e:
        raise OSError(f"File system error saving {dest}: {e}") from e


# --- Core Crawl Logic ---


def crawl(
    clade_qid: str,
    primary_category: str,
    out_dir: Path,
    manifest_csv: Path,
    max_category_depth: int = 2,
    save_interval_images: int = 500, # NEW: Interval for periodic saving
):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, str | int | None]] = []  # Allow None for optional fields
    downloaded_count = 0
    skipped_count = 0

    # Step 1: Get initial taxa and their associated P373 categories from Wikidata
    taxa_info = get_initial_taxa_and_categories(clade_qid)

    # Save the intermediate taxa_info to a JSON file
    taxa_info_path = out_dir / "taxa_categories.json"
    try:
        with open(taxa_info_path, 'w', encoding='utf-8') as f:
            json.dump(taxa_info, f, indent=4)
        logger.info(f"Intermediate taxa and categories saved to: {taxa_info_path}")
    except Exception as e:
        logger.error(f"Failed to save taxa_categories.json: {e}")


    # Combine categories found in Wikidata with the primary category supplied
    EXCLUDE_CATEGORIES = {
        "Cryptomaster behemoth", "Stygnommatidae", "Santinezia", "Sitalcina rothi",
        "Serracutisoma spelaeum", "Bishopella laciniosa", "Sadocus asperatus",
        "Sitalcina californica", "Cranaidae", "Undulus formosus", "Algidia",
        "Holoscotolemon", "Sclerobunus robustus", "Theromaster", "Vonones sayi",
        "Speleonychia sengeri", "Zalmoxoidea", "Texella reyesi", "Megacina schusteri",
        "Sclerobunus", "Maiorerus", "Cosmetinae", "Zalmoxidae", "Lobonychium palpiplus",
        "Sclerobunus idahoensis", "Paranonychus brunneus", "Pachyloidellus goliath",
        "Phalangodes", "Maiorerus randoi", "Erginulus", "Iguapeia melanocephala",
        "Toccolus kuryi", "Zuma tioga", "Sclerobunus cavicolens", "Karamea lobata",
        "Briggsus pacificus", "Paranonychus", "Trojanella serbica", "Zuma acuta",
        "Ventripila", "Gonyleptoidea", "Sitalcina chalona", "Acromares",
        "Gryne perlata", "Grassatores", "Erebomaster flavescens", "Fumontana deprehendor",
        "Kainonychus akamai", "Iandumoema smeagol", "Platymessa victoriae",
        "Yuria pulcra", "Samooidea", "Phalangodidae", "Euepedanus dashdamirovi",
        "Sadocus funestus", "Epedanidae", "Iandumoema", "Assamioidea",
        "Cladonychiidae", "Hendea myersi", "Sadocus", "Toccolus globitarsis",
        "Sitalcina catalina", "Assamiidae", "Cosmetidae", "Lomanius",
        "Epedanoidea", "Samoidae", "Laniatores", "Speleomontia cavernicola",
        "Travuniidae", "Barinas guanenta", "Triregia", "Taito adrik",
        "Sclerobunus skywalkeri", "Triaenonychoidea", "Prostygnus vestitus",
        "Cryptomaster leviathan", "Plistobunus jaegeri", "Podoctidae",
        "Sclerobunus ungulatus", "Theromaster brunneus", "Cryptomastridae",
        "Pachylinae", "Sclerobunus klomax", "Sclerobunus nondimorphicus",
        "Zuma", "Sitalcina", "Algidia chiltoni", "Sitalcina seca", "Bandona",
        "Sclerobunus glorietus", "Pellobunus insularis", "Sitalcina peacheyi",
        "Isolachus spinosus", "Sadocus ingens", "Cynorta dentipes",
        "Algidia homerica", "Stygnus", "Peltonychia leprieurii", "Cryptomaster",
        "Texella bifurcata", "Sadocus dilatatus", "Triaenonychidae",
        "Ampheres spinipes", "Paranonychidae", "Megacina cockerelli",
        "Holoscotolemon lessiniense", "Erebomaster acanthinus", "Gonyleptidae",
        "Poecilaemula", "Acromares vittatum", "Karamea (harvestman)",
        "Erebomaster", "Soerensenella prehensor", "Sadocus polyacanthus",
        "Paecilaema", "Sclerobunus speoventus", "Cynorta", "Texella",
        "Pellobunus", "Briggsus flavescens", "Fumontana", "Pachylospeleus strinatii",
        "Speleomontia", "Sclerobunus madhousensis", "Metanippononychus",
        "Prasma tuberculata", "Briggsus", "Stenostygnellus flavolimbatus",
        "Sitalcina lobata", "Barinas virginis", "Vonones (genus)",
        "Hernandria", "Bishopella", "Sitalcina flava", "Peltonychia",
        "Sitalcina sura", "Flirtea batman", "Lacronia", "Avima wayuunaiki",
        # Adding more general Arachnid or Spider categories found in Commons
        "Arachnida", "Spiders", "Opiliones", "Araneae",
        "Spider photos",
        "Spider images",
    }

    all_categories_to_crawl: Set[str] = {primary_category}
    for taxon_qid, categories in taxa_info.items():
        for cat in categories:
            # Only add categories if they are NOT in our exclusion list
            if cat not in EXCLUDE_CATEGORIES:
                all_categories_to_crawl.add(cat)
            else:
                logger.debug("Skipping excluded category: %s", cat)

    logger.info(
        "Filtered categories identified for crawling: %s", all_categories_to_crawl
    )

    # Step 2A: Collect files from categories
    category_titles: Set[str] = set()
    for cat in all_categories_to_crawl:
        category_titles.update(
            get_files_from_category(cat, max_depth=max_category_depth)
        )
    logger.info(
        "Found %d unique file titles from category crawling.", len(category_titles)
    )

    # Step 2B: Collect files from SDC (P180) and Wikidata (P18)
    sdc_p18_titles: Set[str] = set()
    for taxon_qid in taxa_info:  # Iterate over QIDs found in step 1
        # TEMPORARILY COMMENTING OUT SDC QUERY DUE TO AUTHENTICATION ISSUES
        # sdc_p18_titles.update(get_files_for_taxon_via_sdc(taxon_qid))
        sdc_p18_titles.update(get_files_for_taxon_via_wikidata_p18(taxon_qid))

    # Check if SDC queries were actually executed
    if (
        "get_files_for_taxon_via_sdc" not in globals()
    ):  # A simple check to see if it's commented out
        logger.warning(
            "SDC (P180) queries are currently disabled due to ongoing authentication issues. No files will be fetched from SDC."
        )

    logger.info(
        "Found %d unique file titles from SDC/P18 queries (SDC may be skipped).",
        len(sdc_p18_titles),
    )

    # Combine all found titles and remove duplicates (Sets handle this naturally)
    all_image_titles = category_titles.union(sdc_p18_titles)
    logger.info("Total unique image titles to process: %d", len(all_image_titles))

    # Process images in chunks for imageinfo and download
    titles_list = list(all_image_titles)  # Convert to list for chunking
    if not titles_list:
        logger.info("No images to process. Exiting crawl.")
        return  # Exit if no titles found

    for i in range(
        0, len(titles_list), 50
    ):  # Process 50 titles at a time for imageinfo
        chunk = titles_list[i : i + 50]
        meta_map = fetch_imageinfo(chunk)

        for title, meta in meta_map.items():
            url = meta.get("url")
            if not url:
                logger.debug("No URL found for title: %s", title)
                skipped_count += 1
                continue

            # Basic filtering for common image types (adjust as needed)
            mime_type = meta.get("mime", "").lower()
            # Exclude non-image types like audio (.ogg from your log), svg, and gif, PDF
            if not mime_type.startswith("image/") or mime_type in [
                "image/svg+xml",
                "image/gif",
                "application/pdf",
                "audio/ogg",
            ]:
                logger.debug(
                    "Skipping non-image or unsupported media type (%s): %s",
                    mime_type,
                    title,
                )
                skipped_count += 1
                continue

            # Create a more robust filename to avoid OS issues
            # Replace invalid characters with underscore, preserve alphanumeric, period, dash, underscore
            safe_title = "".join(c if c.isalnum() or c in ".-_" else "_" for c in title)
            # Add original extension if available
            file_extension = Path(url).suffix
            if (
                not file_extension and "." in title
            ):  # Fallback if URL doesn't have a clear extension
                file_extension = Path(title).suffix

            dest_filename = f"{safe_title}{file_extension}"

            image_dest_dir = out_dir / "images"
            dest = image_dest_dir / dest_filename

            try:
                download(url, dest)
                downloaded_count += 1
                em = meta.get("extmetadata", {})
                all_rows.append(
                    {
                        "title": title,
                        "url": url,
                        "sha1": meta.get("sha1"),
                        "mime_type": mime_type,
                        "width": meta.get("width"),
                        "height": meta.get("height"),
                        "size": meta.get("size"),  # Add file size
                        # Use .get with empty dict then .get with default "" for nested fields
                        "license": em.get("LicenseShortName", {}).get("value", ""),
                        "author": em.get("Artist", {}).get("value", ""),
                        "description": em.get("ImageDescription", {}).get(
                            "value", ""
                        ),  # Added description
                        "datetimeoriginal": em.get("DateTimeOriginal", {}).get(
                            "value", ""
                        ),  # Added original datetime
                        "local_path": str(
                            dest.relative_to(out_dir)
                        ),  # Store relative path
                    }
                )
                # NEW: Periodic saving of manifest
                if downloaded_count % save_interval_images == 0:
                    logger.info(f"Periodically saving manifest after {downloaded_count} images...")
                    pd.DataFrame(all_rows).to_csv(manifest_csv, index=False)
                    logger.info("Manifest saved.")

            except (ConnectionError, OSError) as e:  # Catch specific download errors
                logger.warning("Download failed for %s: %s", title, e)
                skipped_count += 1
            except Exception as e:
                logger.error("Unexpected error during processing of %s: %s", title, e)
                skipped_count += 1

    # Final save of the manifest, in case the total count wasn't a multiple of save_interval_images
    if all_rows:
        pd.DataFrame(all_rows).to_csv(manifest_csv, index=False)
        logger.info(
            "Crawl complete. Downloaded %d images. Skipped %d images.",
            downloaded_count,
            skipped_count,
        )
        logger.info("Manifest saved to: %s", manifest_csv)
    else:
        logger.warning("No image data collected to save to manifest.")
        logger.info(
            "Crawl complete. Downloaded %d images. Skipped %d images.",
            downloaded_count,
            skipped_count,
        )


# --- CLI ---
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Crawl Wikimedia Commons for winged insect images using combined strategies."
    )
    ap.add_argument(
        "--clade-qid",
        default="Q2269932",  # Changed to Fulgoridae
        help="Root clade QID (e.g., Q13689 for Insecta, Q746327 for Odonata, Q2269932 for Fulgoridae)",
    )
    ap.add_argument(
        "--primary-category",
        default="Fulgoridae",  # Changed to Fulgoridae
        help="Primary Wikimedia Commons category name to start crawling from (e.g., 'Insects', 'Odonata', 'Fulgoridae')",
    )
    ap.add_argument(
        "--max-category-depth",
        type=int,
        default=2,
        help="Maximum subcategory depth to explore for category-based crawling (default: 2)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to save downloaded images and manifest.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path for the output CSV manifest file.",
    )
    ap.add_argument(
        "--log",
        type=Path,
        default=Path("crawl_commons.log"),
        help="Path for the log file.",
    )
    ap.add_argument(
        "--save-interval",
        type=int,
        default=500, # NEW: Default to saving every 500 images
        help="Number of images after which to periodically save the manifest.csv.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for more verbose output.",
    )
    args = ap.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.log, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # --- IMPORTANT ---
    # Please change the User-Agent contact email to your actual contact info.
    # This is crucial for being a good citizen on Wikimedia Commons.
    logger.info("Starting Wikimedia Commons image crawler...")
    logger.info("Root Clade QID: %s", args.clade_qid)
    logger.info("Primary Commons Category: %s", args.primary_category)
    logger.info("Max Category Depth: %d", args.max_category_depth)
    logger.info("Output Directory: %s", args.out_dir.resolve())
    logger.info("Manifest File: %s", args.manifest.resolve())
    logger.info("Log File: %s", args.log.resolve())
    logger.info("Manifest save interval (images): %d", args.save_interval)

    crawl(
        args.clade_qid,
        args.primary_category,
        args.out_dir,
        args.manifest,
        args.max_category_depth,
        args.save_interval # Pass the new argument
    )
    logger.info("Crawler finished.")


if __name__ == "__main__":
    main()