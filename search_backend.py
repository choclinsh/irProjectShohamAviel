import bz2
import math
import os
import pickle
import re
from collections import Counter, defaultdict
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
from google.cloud import storage
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration & Constants ---
# Stopwords: Union of standard English stopwords and corpus-specific noise
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[#\@\w](['\-]?\w){2,24}""", re.UNICODE)
DB_FILENAME = 'titles.db'
BUCKET_NAME = 'bucket3224'


def read_pkl(base_dir, filename):
    """
    Helper to handle the "Cloud vs Local" dilemma.
    It checks if a file exists locally; if not, it fetches it from the GCP bucket.
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

    # now load and return (Runs for both local-found and just-downloaded)
    print(f"Loading {filename}...")
    with open(path, 'rb') as f:
        return pickle.load(f)


def read_page_views():
    """
    Downloads and processes the huge Wikipedia pageview dump.

    Optimization:
    - Uses streaming (iter_content) for the download to avoid memory spikes.
    - filters lines *immediately* while reading the compressed bz2 file to keep the
      memory footprint low (only storing en.wikipedia articles).
    """
    # Setup Paths
    pv_path = 'https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2'
    filename = 'pageviews-202108-user.bz2'
    pickle_filename = 'pageviews-202108-user.pkl'

    # Check if pickle already exists
    if os.path.exists(pickle_filename):
        print(f"Loading pageviews from {pickle_filename}...")
        with open(pickle_filename, 'rb') as f:
            wid2pv = pickle.load(f)
            return wid2pv

    # Download the bz2 file if it doesn't exist
    if not os.path.exists(filename):
        print(f"Downloading {filename} ")
        response = requests.get(pv_path, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    print("Processing pageviews (streaming and filtering)...")
    wid2pv = Counter()

    line_count = 0

    # Open the compressed file directly
    with bz2.open(filename, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_count += 1
            parts = line.split(' ')

            # Filter: strict check for 'en.wikipedia' and valid IDs
            if len(parts) > 4 and parts[0] == 'en.wikipedia':
                doc_id_str = parts[2]
                views_str = parts[4]

                if doc_id_str.isdigit() and views_str.isdigit():
                    doc_id = int(doc_id_str)
                    views = int(views_str)
                    wid2pv[doc_id] += views

    # Save the result to pickle for next time
    print(f"Saving processed data to {pickle_filename}...")
    with open(pickle_filename, 'wb') as f:
        pickle.dump(wid2pv, f)

    return wid2pv


# --- Initialization ---
# Define english stopwords from nltk Oct 2025
english_stopwords = frozenset([
    "during", "as", "whom", "no", "so", "shouldn't", "she's", "were", "needn", "then", "on",
    "should've", "once", "very", "any", "they've", "it's", "it", "be", "why", "ma", "over",
    "you'll", "they", "you've", "am", "before", "shan", "nor", "she'd", "because", "been",
    "doesn't", "than", "will", "they'd", "not", "those", "had", "this", "through", "again",
    "ours", "having", "himself", "into", "i'm", "did", "hadn", "haven", "should", "above",
    "we've", "does", "now", "m", "down", "he'd", "herself", "t", "their", "hasn't", "few",
    "and", "mightn't", "some", "do", "the", "we're", "myself", "i'd", "won", "after",
    "needn't", "wasn't", "them", "don", "further", "we'll", "hasn", "haven't", "out", "where",
    "mustn't", "won't", "at", "against", "shan't", "has", "all", "s", "being", "he'll", "he",
    "its", "that", "more", "by", "who", "i've", "o", "that'll", "there", "too", "they'll",
    "own", "aren't", "other", "an", "here", "between", "hadn't", "isn't", "below", "yourselves",
    "ve", "isn", "wouldn", "d", "we", "couldn", "ain", "his", "wouldn't", "was", "didn", "what",
    "when", "i", "i'll", "with", "her", "same", "you're", "yours", "couldn't", "for", "doing",
    "each", "aren", "which", "such", "mightn", "up", "mustn", "you", "only", "most", "of", "me",
    "she", "he's", "in", "a", "if", "but", "these", "him", "hers", "both", "my", "she'll", "re",
    "weren", "yourself", "is", "until", "weren't", "to", "are", "itself", "you'd", "themselves",
    "ourselves", "just", "wasn", "have", "don't", "ll", "how", "they're", "about", "shouldn",
    "can", "our", "we'd", "from", "it'd", "under", "while", "off", "y", "doesn", "theirs",
    "didn't", "or", "your", "it'll"
])

all_stopwords = english_stopwords.union(corpus_stopwords)

# Pre-load global data structures
DL = read_pkl("postings_gcp", "dl_body.pkl")
page_views_dict = read_page_views()

# Log-normalization of page views to prevent popularity from dominating relevance
pv_scores = {
    doc_id: 1 + math.log10(1 + count * 0.1)
    for doc_id, count in page_views_dict.items()
}


def search(query, body_index, titles_index, anchor_index):
    '''
    Main entry point for the search engine.

    Architecture:
    1. Parallel execution: Title, Anchor, and Body searches run in threads to reduce latency.
    2. Merge: Results are weighted and combined.
    3. Retrieve Titles: Titles are fetched from SQLite in a single batch

    Returns:
        list of (wiki_id, title) tuples.
    '''

    # Use ThreadPool to run independent searches in parallel
    with ThreadPoolExecutor() as executor:
        # Submit tasks
        future_title = executor.submit(distinct_words_logic_scores, query, titles_index)
        future_anchor = executor.submit(distinct_words_logic_scores, query, anchor_index)
        future_body = executor.submit(body_index.get_bm, query)

        # Get results
        title_scores = future_title.result()
        anchor_scores = future_anchor.result()
        body_scores = future_body.result()

    # Optimization: Filter top N *before* merging to reduce dictionary operations.
    # We keep a larger candidate pool (2000) for merging to avoid missing documents
    # that might have moderate scores in all three categories.
    title_scores = get_top_n(title_scores, 2000)
    anchor_scores = get_top_n(anchor_scores, 2000)
    # body_scores are already sorted/filtered by get_bm

    merged_scores = merge_results(title_scores, body_scores, anchor_scores)
    top_scores = get_top_n(merged_scores, 100)

    # Extract IDs for DB lookup
    top_doc_ids = [doc_id for doc_id, score in top_scores]
    titles_map = get_titles_batch(top_doc_ids)

    # Join the sorted IDs with their titles from the map
    return [(str(doc_id), titles_map.get(doc_id, "Title Not Found")) for doc_id in top_doc_ids]


def merge_results(title_scores, body_scores, anchor_scores, title_weight=0.4, text_weight=0.3, anchor_weight=0.3,
                  N=100):
    """
    Combines scores from different indices using a weighted average.

    Strategy:
    - Titles are heavily weighted (high precision).
    - 'Extra' points are added to Title/Anchor matches to boost documents where the query
      appears in the metadata, not just the text.
    - PageView scores are applied as a multiplicative factor to boost popular pages.
    """
    all_scores = defaultdict(float)
    title_extra = 10
    anchor_extra = 5

    # Add Weighted Title Scores
    for doc_id, score in title_scores:
        all_scores[doc_id] += (score + title_extra) * title_weight

    # Add Weighted Body Scores
    for doc_id, score in body_scores:
        all_scores[doc_id] += score * text_weight

    # Add Weighted Anchor Scores
    for doc_id, score in anchor_scores:
        all_scores[doc_id] += (score + anchor_extra) * anchor_weight

    # Convert to list of pairs and apply PageView impact
    merged_list = [(doc_id, score * pv_scores.get(doc_id, 1)) for doc_id, score in all_scores.items()]

    # Sort by score descending and take Top N
    return sorted(merged_list, key=lambda x: x[1], reverse=True)[:N]


def search_distinct_words_logic(query, index):
    """
    Wrapper for binary/count-based search (Title/Anchor).
    Fetches titles for the results before returning.
    """
    sorted_pairs = distinct_words_logic_scores(query, index)
    sorted_ids = [doc_id for doc_id, score in sorted_pairs]

    # Fetch ALL titles in one (or few) fast SQL queries
    # This replaces the loop that called get_title() thousands of times
    titles_map = get_titles_batch(sorted_ids)

    sorted_pairs = [(str(doc_id), titles_map.get(doc_id, "Title Not Found")) for doc_id in sorted_ids]

    return sorted_pairs


def distinct_words_logic_scores(query, index):
    """
    Calculates scores based on how many distinct query words appear in the document.
    Used primarily for Title and Anchor text where TF-IDF is less effective due to short length.
    """
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    # filter stopwords and keep only UNIQUE tokens
    valid_tokens = set(token for token in tokens if token not in all_stopwords)
    distinct_words_dict = Counter()

    for token in valid_tokens:
        try:
            tokens_pl = index.read_a_posting_list("", token)
            for pair in tokens_pl:
                doc_id = pair[0]
                distinct_words_dict[doc_id] += 1  # increase the value by 1 for each doc that appear in the terms pl
        except:
            # if token is not in index, skip it
            continue

    sorted_docs = distinct_words_dict.most_common()

    if not sorted_docs:
        return []

    return sorted_docs


def search_body_logic(query, index):
    '''
    Wrapper for Body search (Cosine Similarity / TF-IDF).
    '''
    top_scores = search_body_logic_scores(query, index)
    top_doc_ids = [doc_id for doc_id, score in top_scores]
    titles_map = get_titles_batch(top_doc_ids)

    return [(str(doc_id), titles_map.get(doc_id, "Title Not Found")) for doc_id in top_doc_ids]


def search_body_logic_scores(query, index):
    '''
    Standard Vector Space Model search.
    Generates TF-IDF vectors for the query and candidate documents, then calculates Cosine Similarity.
    '''
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    # filter stopwords and remove duplicates
    valid_query = set(token for token in tokens if token not in all_stopwords)

    query_vector = generate_query_tfidf_vector(valid_query, index)

    if np.all(query_vector == 0):  # no document fit if every vector in query is 0
        return []

    pls = []
    for w in valid_query:  # save postings list only for terms in the query
        pls.append(index.read_a_posting_list("postings_gcp", w))

    D = generate_document_tfidf_matrix(valid_query, index, list(valid_query), pls)

    if D.empty:
        return []

    cos_sim_dict = cosine_similarity_sklearn(D, query_vector)

    top_scores = get_top_n(cos_sim_dict.items(), 100)
    if not top_scores:
        return []
    return top_scores


def get_titles_batch(doc_ids):
    """
    Performance Fix: Fetches titles in batches.
    SQLite has a variable limit (often 999), so we chunk requests to avoid crashing.
    This reduces overhead significantly compared to individual lookups.
    """
    if not doc_ids:
        return {}

    chunk_size = 500
    id_list = list(doc_ids)
    results_map = {}

    try:
        with sqlite3.connect(DB_FILENAME) as conn:
            cursor = conn.cursor()

            # Process in chunks of 500
            for i in range(0, len(id_list), chunk_size):
                chunk = id_list[i: i + chunk_size]
                # Create a string of placeholders like "?, ?, ?"
                placeholders = ','.join(['?'] * len(chunk))
                query = f"SELECT id, title FROM documents WHERE id IN ({placeholders})"

                cursor.execute(query, chunk)

                # Store results in a dictionary for O(1) lookup later
                for doc_id, title in cursor.fetchall():
                    results_map[doc_id] = title

    except Exception as e:
        print(f"Database error in batch lookup: {e}")

    return results_map


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generates a TF-IDF vector for the query.
    Note: Uses log base 10 for IDF and normalizes TF by query length.
    """

    epsilon = .0000001
    Q = np.zeros((len(query_to_search)))

    counter = Counter(query_to_search)

    for i, token in enumerate(query_to_search):
        if token in index.term_total:  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)
            df = index.df[token]
            idf = math.log((len(DL)) / (df + epsilon), 10)  # smoothing
            Q[i] = tf * idf
    return Q


def get_candidate_documents_and_scores(query_to_search, index, words, pls, max_docs=1000):
    """
    First pass: Identify candidate documents that contain query terms.
    Calculates partial TF-IDF scores during the pass.
    """
    candidates = {}
    for term in query_to_search:
        if term in words:
            list_of_doc = pls[words.index(term)]

            if len(list_of_doc) > max_docs:
                list_of_doc = sorted(list_of_doc, key=lambda x: x[1], reverse=True)[:max_docs]

            normlized_tfidf = [(doc_id, (freq / DL[doc_id]) * math.log(len(DL) / index.df[term], 10)) for doc_id, freq
                               in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Constructs the Document-Term matrix (DataFrame) for cosine similarity.
    Rows = Doc IDs, Columns = Query Terms.
    """

    vocab_size = len(words)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words, pls)

    # Only utilize documents which have corresponding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = words

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        if term in words:
            D.loc[doc_id, term] = tfidf

    return D


def cosine_similarity_sklearn(D, Q):
    """
    Wraps sklearn's cosine_similarity for our DataFrame structure.
    Returns a dictionary mapping {doc_id: score}.
    """
    # we reshape Q to (1, -1) because sklearn expects a 2D array
    scores = cosine_similarity(D, Q.reshape(1, -1))

    # flatten it to a simple 1D array.
    scores_flat = scores.flatten()

    # zip the existing DataFrame indices with the new scores
    return dict(zip(D.index, scores_flat))


def get_top_n(score_pairs, N=3):
    """
    Sorts and trims the list of (doc_id, score) tuples.
    Rounds scores to 5 decimal places for cleanliness.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in score_pairs if score > 0], key=lambda x: x[1],
                  reverse=True)[:N]


class BM25_from_index:
    """
    Best Match 25 (BM25) Implementation.
    A probabilistic ranking function used as the primary scorer for Body text.

    Parameters:
    - k1: Controls term saturation (1.2-2.0 usually)
    - b: Controls length normalization (0.75 usually)
    """

    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N

    def calc_idf(self, list_of_tokens):
        """
        Calculates IDF with standard smoothing.
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def get_bm(self, query, N=1000):
        """
        Calculates BM25 scores for the given query.
        OPTIMIZED for speed by using dictionary lookups.
        """
        query_bm25scores = []
        tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        valid_query = set(token for token in tokens if token not in all_stopwords)
        ordered_tokens = list(valid_query)

        # 1. Pre-calculate IDF
        idf_dict = self.calc_idf(ordered_tokens)

        # 2. Retrieve PLs and convert to Dicts ONCE
        # This prevents re-iterating over posting lists for every document.
        pl_dicts = []
        candidate_docs = set()

        for w in valid_query:
            if idf_dict[w] < 0.1:  # Skip very common words (stopword-like) to save time
                pl_dicts.append({})
                continue

            # Read the list of tuples [(doc_id, freq), ...]
            pl = self.index.read_a_posting_list("postings_gcp", w)

            # Convert to dictionary immediately: {doc_id: freq} for O(1) access
            pl_dict = dict(pl)

            pl_dicts.append(pl_dict)
            candidate_docs.update(pl_dict.keys())

        # Calculate BM25 for each candidate
        for doc_id in candidate_docs:
            score = self._score(ordered_tokens, doc_id, pl_dicts, idf_dict)
            query_bm25scores.append((doc_id, score))

        return sorted(query_bm25scores, key=lambda x: x[1], reverse=True)[:N]

    def _score(self, ordered_tokens, doc_id, pl_dicts, idf_dict):
        """
        Calculates score for a single document using the pre-built posting list dictionaries.
        """
        score = 0.0

        if doc_id not in DL:
            return 0.0

        doc_len = DL[doc_id]

        for i, term in enumerate(ordered_tokens):
            # Optimization: Check existence in the term's dictionary (O(1))
            if doc_id in pl_dicts[i]:
                freq = pl_dicts[i][doc_id]
                numerator = idf_dict[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                score += (numerator / denominator)

        return score


def get_pageview(wiki_ids):
    """ Helper to retrieve pageviews for specific IDs in the same oder they appear
        Parameters:
        -----------
        wiki_ids: list of ints that represent the doc_ids
    """
    views_list = []

    for id in wiki_ids:
        views_list.append(page_views_dict.get(id, 0))

    return views_list
