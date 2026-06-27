"""
Computes sentence embedding features from paper abstracts and saves them to disk.
Replaces the original TF-IDF vectorizer with all-MiniLM-L6-v2 embeddings.
The saved feature dict is backward-compatible with serve.py's load_features() call:
- 'pids': list of paper ids (unchanged)
- 'x': np.ndarray of shape (n_papers, 384), float32, L2-normalized (replaces sparse tfidf matrix)
- 'vocab': empty dict (satisfies serve.py /inspect endpoint without crashing)
- 'idf': empty np.ndarray (satisfies serve.py /inspect endpoint without crashing)
"""

import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from aslite.db import get_papers_db, save_features

EMBED_MODEL = 'all-MiniLM-L6-v2'
BATCH_SIZE = 64  # safe for Pi 4 4GB; reduce to 32 if OOM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arxiv Embedding Computor')
    parser.add_argument('--max_docs', type=int, default=-1,
                        help='maximum number of documents to embed, or -1 for all')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='batch size for embedding inference')
    args = parser.parse_args()
    print(args)

    print(f"loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    pdb = get_papers_db(flag='r')
    all_pids = list(pdb.keys())

    if args.max_docs > 0:
        all_pids = all_pids[:args.max_docs]

    print(f"building corpus for {len(all_pids)} papers...")
    corpus = []
    for pid in all_pids:
        d = pdb[pid]
        author_str = ' '.join([a['name'] for a in d['authors']])
        text = ' '.join([d['title'], d['summary'], author_str])
        corpus.append(text)

    print("encoding embeddings (this will be slow on first run — model downloads ~90MB)...")
    embeddings = model.encode(
        corpus,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize so cosine sim == dot product
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    print(f"embeddings shape: {embeddings.shape}")

    print("saving features to disk...")
    features = {
        'pids': all_pids,
        'x': embeddings,            # shape (n, 384), dense float32
        'vocab': {},                # stub: keeps /inspect from crashing
        'idf': np.array([]),        # stub: keeps /inspect from crashing
    }
    save_features(features)
    print("done.")
