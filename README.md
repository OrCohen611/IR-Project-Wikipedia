# Wikipedia Search Engine

A robust search engine for English Wikipedia articles, integrating content-based retrieval with popularity metrics.

## Project Components
* **Indexing**: Inverted indexes for document body, titles, and anchor text built using PySpark on GCP.
* **Ranking**: Hybrid scoring model combining TF-IDF relevance with PageRank and PageView data.
* **Frontend**: Flask-based REST API that retrieves data from Google Cloud Storage buckets.

## Key Endpoints
1. `/search`: Best-performing search combining title, body, and popularity signals.
2. `/search_body`: Retrieval based on TF-IDF and Cosine Similarity.
3. `/search_title`: Ranking by distinct query term overlap in article titles.
4. `/get_pagerank` / `/get_pageview`: Metric retrieval for specific article IDs.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run server:
```bash
python3 search_frontend.py
```
The server runs on port 8080.
