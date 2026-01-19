from flask import Flask, request, jsonify
import search_backend
import inverted_index_gcp
import os
import pickle
import subprocess
from google.cloud import storage


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

BUCKET_NAME = 'bucket3224'
DB_FILENAME = 'titles.db'


def download_db_if_needed():
    """ Downloads the SQLite database from GCP bucket if not found locally. """
    if os.path.exists(DB_FILENAME):
        print(f"{DB_FILENAME} found locally.")
        return

    print(f"Downloading {DB_FILENAME} from bucket {BUCKET_NAME}...")
    try:  # didnt find it so connect to the bucket and download specific file
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(DB_FILENAME)
        blob.download_to_filename(DB_FILENAME)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading database: {e}")


def read_pkl(base_dir, filename):
    """ Downloads the file from GCP bucket if not found locally.
        Based of the base directory and filename + suffix

        Parameters:
        -----------
        base_dir: str (local directory like "postings_gcp")
        filename: str (like "index.pkl")
    """
    path = os.path.join(base_dir, filename)

    # If not found locally, download it
    if not os.path.exists(path):
        print(f"Downloading {filename} from bucket {BUCKET_NAME}...")
        try:
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(path)
            blob.download_to_filename(path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return {}  # Return empty dict on failure to prevent crash

    # Now load and return (Runs for both local-found and just-downloaded)
    print(f"Loading {filename}...")
    with open(path, 'rb') as f:
        return pickle.load(f)  # return the loaded file info


def download_index_if_needed(base_dir, bucket_name, source_prefix):
    """
    Downloads the entire index folder (pkl + bin files) from the bucket
    using gsutil.

    Parameters:
    -----------
    base_dir: str (local directory name, e.g., 'postings_gcp')
    bucket_name: str (e.g., 'bucket3224')
    source_prefix: str (path inside bucket, e.g., 'postings_gcp')
    """
    # Check if the directory already exists and looks populated
    if os.path.exists(os.path.join(base_dir, "index.pkl")):
        print(f"{base_dir} already exists. Skipping download.")
        return

    print(f"Downloading index from gs://{bucket_name}/{source_prefix} to {base_dir}...")

    # Create the directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # command for faster downloading
    command = [
        "gsutil", "-m", "cp", "-r",
        f"gs://{bucket_name}/{source_prefix}/*",
        f"{base_dir}/"
    ]

    try:
        subprocess.check_call(command)
        print("[SUCCESS] Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download index: {e}")


download_db_if_needed()

print("Loading Title Index...")
download_index_if_needed(base_dir="postings_gcp_title", bucket_name=BUCKET_NAME, source_prefix="postings_gcp_title")
title_index = inverted_index_gcp.InvertedIndex.read_index("postings_gcp_title", "index")  # init index
                                                                                                     # from the file
print("Loading Body Index...")
download_index_if_needed(base_dir="postings_gcp", bucket_name=BUCKET_NAME, source_prefix="postings_gcp")
body_index = inverted_index_gcp.InvertedIndex.read_index("postings_gcp", "index")

body_index.term_total = read_pkl("postings_gcp", "term_total.pkl")  # reading separated term total

body_bm25 = search_backend.BM25_from_index(body_index)  # creating instance to compute bm25

print("Loading Anchor Index...")
download_index_if_needed(base_dir="postings_gcp_anchor", bucket_name=BUCKET_NAME, source_prefix="postings_gcp_anchor")
anchor_index = inverted_index_gcp.InvertedIndex.read_index("postings_gcp_anchor", "index")


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
    res = search_backend.search(query, body_bm25, title_index, anchor_index)  # passing bm25 because the logic needs it
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
    res = search_backend.search_body_logic(query, body_index)
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
    res = search_backend.search_distinct_words_logic(query, title_index)
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
    res = search_backend.search_distinct_words_logic(query, anchor_index)
    # END SOLUTION
    return jsonify(res)


# @app.route("/get_pagerank", methods=['POST'])
# def get_pagerank():
#     ''' Returns PageRank values for a list of provided wiki article IDs.
#
#         Test this by issuing a POST request to a URL like:
#           http://YOUR_SERVER_DOMAIN/get_pagerank
#         with a json payload of the list of article ids. In python do:
#           import requests
#           requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
#         As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
#         if you're using ngrok on Colab or your external IP on GCP.
#     Returns:
#     --------
#         list of floats:
#           list of PageRank scores that correrspond to the provided article IDs.
#     '''
#     res = []
#     wiki_ids = request.get_json()
#     if len(wiki_ids) == 0:
#       return jsonify(res)
#     # BEGIN SOLUTION
#
#     # END SOLUTION
#     return jsonify(res)

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
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = search_backend.get_pageview(wiki_ids)
    # END SOLUTION
    return jsonify(res)


def run(**options):
    app.run(**options)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False, threaded=True)
