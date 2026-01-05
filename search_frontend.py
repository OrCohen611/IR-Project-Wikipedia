from flask import Flask, request, jsonify
import hashlib
import math
import pickle
import re
from collections import Counter, defaultdict
from google.cloud import storage
from inverted_index_gcp import InvertedIndex

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

client = storage.Client()
bucket_name = 'exercise_bucket_ir'
bucket = client.get_bucket(bucket_name)

def load_pickle(file_name):
    blob = bucket.blob(file_name)
    with blob.open("rb") as f:
        return pickle.load(f)

index_body = load_pickle('index_body.pkl')
index_title = load_pickle('index_title.pkl')
index_anchor = load_pickle('index_anchor.pkl')
pagerank_dict = load_pickle('pagerank_dict.pkl')
pageviews_dict = load_pickle('pageviews_dict.pkl')
id_title_dict = load_pickle('id_title_dict.pkl')

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]

def get_posting_list(index, word, bucket_name):
    if word not in index.posting_locs:
        return []
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    locs = index.posting_locs[word]
    posting_list = []
    num_bytes = index.df[word] * 6
    for f_name, offset in locs:
        blob = bucket.blob(f"postings_gcp/{f_name}")
        content = blob.download_as_bytes(start=offset, end=offset + num_bytes - 1)
        for i in range(index.df[word]):
            b = content[i * 6 : (i + 1) * 6]
            doc_id = int.from_bytes(b[:4], 'big')
            tf = int.from_bytes(b[4:], 'big')
            posting_list.append((doc_id, tf))
    return posting_list

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = [t.lower() for t in tokenize(query)]
    scores = defaultdict(float)
    for token in tokens:
        if token in index_title.df:
            try:
                pl = get_posting_list(index_title, token, bucket_name)
                for doc_id, tf in pl:
                    scores[int(doc_id)] += 1000.0
            except:
                continue
        if token in index_body.df:
            try:
                pl = get_posting_list(index_body, token, bucket_name)
                for doc_id, tf in pl:
                    scores[int(doc_id)] += (1 + math.log10(tf))
            except:
                continue

    sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
    res = [(str(doc_id), str(id_title_dict.get(doc_id, f"ID:{doc_id}"))) for doc_id, _ in sorted_res]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)

    query_counts = Counter(tokens)
    scores = defaultdict(float)
    query_norm_sq = 0
    num_docs = len(id_title_dict)

    for token, q_tf in query_counts.items():
        if token in index_body.df:
            idf = math.log10(num_docs / index_body.df[token])
            q_tfidf = q_tf * idf
            query_norm_sq += q_tfidf ** 2
            pl = get_posting_list(index_body, token, bucket_name)
            for doc_id, tf in pl:
                scores[doc_id] += (tf * idf) * q_tfidf

    if not scores: return jsonify([])

    query_norm = math.sqrt(query_norm_sq)
    results = []
    for doc_id, score in scores.items():
        doc_norm = index_body.norms.get(doc_id, 1.0)
        normalized_score = score / (doc_norm * query_norm)
        results.append((doc_id, normalized_score))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:100]
    res = [(str(doc_id), id_title_dict.get(doc_id, "Title not found")) for doc_id, score in sorted_results]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = set(tokenize(query))
    doc_matches = defaultdict(int)
    for token in tokens:
        if token in index_title.df:
            pl = get_posting_list(index_title, token, bucket_name)
            for doc_id, tf in pl:
                doc_matches[doc_id] += 1

    sorted_results = sorted(doc_matches.items(), key=lambda x: x[1], reverse=True)
    res = [(str(doc_id), id_title_dict.get(doc_id, "Title not found")) for doc_id, count in sorted_results]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = set(tokenize(query))
    doc_matches = defaultdict(int)
    for token in tokens:
        if token in index_anchor.df:
            pl = get_posting_list(index_anchor, token, bucket_name)
            for doc_id, tf in pl:
                doc_matches[doc_id] += 1

    sorted_results = sorted(doc_matches.items(), key=lambda x: x[1], reverse=True)
    res = [(str(doc_id), id_title_dict.get(doc_id, "Title not found")) for doc_id, count in sorted_results]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if not wiki_ids or len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [pagerank_dict.get(int(doc_id), 0) for doc_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if not wiki_ids or len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [pageviews_dict.get(int(doc_id), 0) for doc_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)
    
if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)