import os
import math
import pandas as pd
import string
from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def normalize_text(text):
    if pd.isnull(text):
        return []
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return [stemmer.stem(w) for w in tokens if w not in stop_words]

def build_index():
    if not os.path.exists("documents"):
        print("Missing 'documents' folder!")
        return

    doc_term_freqs = []
    doc_lengths = []
    term_doc_counts = defaultdict(int)
    all_docs = []
    doc_id = 0

    for filename in os.listdir("documents"):
        if filename.endswith((".txt", ".csv")):
            path = os.path.join("documents", filename)
            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(path, encoding='utf-8', errors='ignore')
                    if 'text' not in df.columns:
                        continue
                    texts = df['text']
                else:
                    with open(path, encoding='utf-8', errors='ignore') as f:
                        texts = [f.read()]
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

            for text in texts:
                tokens = normalize_text(text)
                tf = Counter(tokens)
                doc_lengths.append(len(tokens))
                doc_term_freqs.append(tf)
                for term in tf:
                    term_doc_counts[term] += 1
                all_docs.append(f"{filename}#doc{doc_id}")
                doc_id += 1

    N = len(doc_term_freqs)
    idf = {term: math.log(N / df) for term, df in term_doc_counts.items()}
    indexed_data = []
    for i, tf in enumerate(doc_term_freqs):
        for term, freq in tf.items():
            weight = freq * idf[term]
            indexed_data.append({
                'Document': all_docs[i],
                'Term': term,
                'Term Frequency': freq,
                'Document Length': doc_lengths[i],
                'Inverse Document Frequency': round(idf[term], 4),
                'TF-IDF Weight': round(weight, 4)
            })

    pd.DataFrame(indexed_data).to_csv('text_index.csv', index=False)

build_index()

def cosine_similarity(vec1, vec2, vocab):
    dot = sum(vec1.get(t, 0.0) * vec2.get(t, 0.0) for t in vocab)
    norm1 = math.sqrt(sum(v*v for v in vec1.values()))
    norm2 = math.sqrt(sum(v*v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        terms = normalize_text(query)
        query_tf = Counter(terms)

        if not query_tf:
            return render_template("results.html",
                                   query=query,
                                   results=[],
                                   message="Your query contains no valid terms.")

        index_df = pd.read_csv("text_index.csv")
        docs = index_df['Document'].unique()
        doc_vectors = defaultdict(dict)
        vocab = set(index_df['Term'])

        for _, row in index_df.iterrows():
            doc_vectors[row['Document']][row['Term']] = row['TF-IDF Weight']

        N = len(docs)
        # build idf for query terms (0 if term never appears)
        idf = {}
        for term in query_tf:
            df_count = sum(1 for d in doc_vectors if term in doc_vectors[d])
            idf[term] = math.log(N/df_count) if df_count > 0 else 0.0

        # build query vector
        query_vec = {term: query_tf[term] * idf[term] for term in query_tf}

        # score every document
        scores = []
        for doc in docs:
            sim = cosine_similarity(query_vec, doc_vectors.get(doc, {}), vocab)
            scores.append((doc, round(sim,4)))

        # sort descending, keep all including zeros
        results = sorted(scores, key=lambda x: x[1], reverse=True)

        message = None
        if all(score == 0.0 for _, score in results):
            message = "No matching terms found in any document."

        return render_template("results.html",
                               query=query,
                               results=results,
                               message=message)

    return render_template("index.html")

# — this route must exist if your template uses url_for('view_document', doc_id=...)
@app.route("/document/<path:doc_id>")
def view_document(doc_id):
    filename = doc_id.split("#")[0]
    path = os.path.join("documents", filename)
    try:
        if filename.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        elif filename.endswith(".csv"):
            df = pd.read_csv(path, encoding='utf-8', errors='ignore')
            content = df.to_string()
        else:
            content = "Unsupported file type."
    except Exception as e:
        content = f"Error reading file: {e}"
    return f"<pre>{content}</pre>"

@app.route("/index")
def show_index():
    df = pd.read_csv("text_index.csv", encoding='utf-8', engine='python')
    return df.to_html(classes="table table-striped table-bordered", index=False)

if __name__ == "__main__":
    app.run(debug=True)