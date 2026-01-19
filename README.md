# Search Engine Project

## Overview

This project implements a scalable search engine for English Wikipedia articles, designed for deployment on Google Cloud Platform (GCP). It features a Flask-based frontend for handling API requests and a high-performance backend that utilizes inverted indices, efficient SQL querying, and multi-threaded processing to deliver low-latency search results.

## Code Structure & Organization

The codebase is organized into two primary Python files and utilizes external data assets (indices, SQLite databases) managed via Google Cloud Storage.

### 1. `search_frontend.py`

This script serves as the HTTP entry point for the application. It handles server configuration, data loading, and request routing.

- **Initialization & Data Loading:**
  - **Automated Asset Retrieval:** Checks for local existence of necessary files (databases, indices). If missing, it automatically downloads them from the GCP bucket (`bucket3224`) using `google.cloud.storage` or `gsutil` for bulk transfers.
  - **Inverted Index Loading:** Loads the Title, Body, and Anchor inverted indices into memory upon startup to ensure fast access.
  - **Scorer Initialization:** Pre-calculates global statistics (e.g., `term_total`) and initializes the BM25 scoring instance for the body index.

- **API Endpoints:**
  - `/search`: The primary endpoint. Accepts a query and returns the top 100 results using a weighted combination of BM25 (Body) and binary/count-based scoring (Title/Anchor).
  - `/search_body`: Returns results ranked solely by Cosine Similarity using TF-IDF on the body text.
  - `/search_title` & `/search_anchor`: Returns results based on matches in the article title or anchor text, ranked by the count of distinct query terms found.
  - `/get_pageview`: Helper endpoint to retrieve August 2021 page view counts for a list of document IDs.

### 2. `search_backend.py`

This module contains the core information retrieval logic, ranking algorithms, and data processing utilities.

- **Search Logic (`search` function):**
  - **Parallel Execution:** Utilizes `ThreadPoolExecutor` to query the Title, Anchor, and Body indices concurrently, significantly reducing wall-clock latency.
  - **Result Merging:** Combines scores from the three indices using a weighted average (Title: 0.4, Body: 0.3, Anchor: 0.3). Matches in high-value fields (Title/Anchor) are given additional boost points.
  - **Popularity Re-ranking:** Incorporates PageView data to adjust final scores. To prevent popularity from overwhelming textual relevance, page views are log-normalized (`1 + log10(views)`).

- **Ranking Algorithms:**
  - **BM25 (`BM25_from_index`):** The primary ranking function for body text. It is optimized to read only relevant posting lists and converts them to dictionaries for $O(1)$ lookup speed during scoring.
  - **Cosine Similarity:** Implements a Vector Space Model using TF-IDF and `sklearn`'s cosine similarity for the `/search_body` endpoint.
  - **Binary/Count Ranking:** Used for Title and Anchor searches, ranking documents by the number of distinct query terms they contain.

- **Data Management:**
  - **`get_titles_batch`:** An optimized SQL retriever. Instead of performing individual DB lookups, it fetches document titles in batches (e.g., chunks of 500) using `WHERE id IN (...)` clauses to minimize overhead.
  - **`read_page_views`:** Parses the compressed Wikipedia pageview dump, filtering for English Wikipedia entries via stream processing to manage memory usage.

## Usage

You can use the server with this query for example: "http://34.170.40.177:8080/search?query=information+retrieval"
34.170.40.177 is the external ip of our instance.

## Dependencies

Flask: Web server framework.

Google Cloud Storage: For downloading indices and databases.

Pandas / NumPy / Scikit-Learn: For vector operations and matrix calculations.

SQLite3: For fast metadata (title) retrieval.
