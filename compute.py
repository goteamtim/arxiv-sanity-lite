"""
Computes sentence embedding features from paper abstracts and saves them to disk.
Uses fastembed (ONNX-based, no PyTorch) with all-MiniLM-L6-v2 for CPU-only inference
on low-memory hardware like Raspberry Pi.
The saved feature dict is backward-compatible with serve.py's load_features() call:
- 'pids': list of paper ids (unchanged)
- 'x': np.ndarray of shape (n_papers, 384), float32, L2-normalized
- 'vocab': empty dict (satisfies serve.py /inspect endpoint without crashing)
- 'idf': empty np.ndarray (satisfies serve.py /inspect endpoint without crashing)
"""

import argparse
import numpy as np
from fastembed import TextEmbedding
from aslite.db import get_papers_db, save_features

EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 32  # conservative default for low-memory hardware

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arxiv Embedding Computor')
    parser.add_argument('--max_docs', type=int, default=-1,
                        help='maximum number of documents to embed, or -1 for all')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='batch size for embedding inference')
    args = parser.parse_args()
    print(args)

    print(f"loading embedding model: {EMBED_MODEL}")
    model = TextEmbedding(model_name=EMBED_MODEL)

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

    print("encoding embeddings (first run downloads ~90MB model)...")
    embeddings = np.array(list(model.embed(corpus, batch_size=args.batch_size)), dtype=np.float32)
    print(f"embeddings shape: {embeddings.shape}")

    print("saving features to disk...")
    features = {
        'pids': all_pids,
        'x': embeddings,
        'vocab': {},
        'idf': np.array([]),
    }
    save_features(features)
    print("done.")
