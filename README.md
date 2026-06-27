
# arxiv-sanity-lite

A much lighter-weight arxiv-sanity from-scratch re-write. Periodically polls the arxiv API for new papers, then allows users to tag papers of interest and recommends new papers for each tag using SVMs over sentence embeddings of paper abstracts. Allows one to search, rank, sort, slice and dice these results in a pretty web UI. Lastly, arxiv-sanity-lite can send you daily emails with recommendations of new papers based on your tags. Curate your tags, track recent papers in your area, and don't miss out!

I am running a live version of this code on [arxiv-sanity-lite.com](https://arxiv-sanity-lite.com).

![Screenshot](screenshot.jpg)

#### Requirements

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install uv, then:

```bash
uv sync
```

This creates a `.venv` and installs all dependencies including PyTorch and `sentence-transformers`.

#### To run

Fetch papers from the arxiv API (run this periodically, e.g. via cron):

```bash
uv run python arxiv_daemon.py --num 2000
```

Compute sentence embeddings for all papers:

```bash
uv run python compute.py
```

The first run downloads the `all-MiniLM-L6-v2` model (~90MB). On a Raspberry Pi 4, embedding ~2000 papers takes roughly 5–10 minutes. Subsequent runs are faster since the model is cached.

Serve the Flask app locally:

```bash
uv run flask --app serve run
```

Or use the Makefile shortcuts:

```bash
make up   # fetch new papers + recompute embeddings
make fun  # start the dev server
```

All database files are stored in the `data/` directory.

#### Embeddings

Paper recommendations are powered by [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) instead of TF-IDF. Each paper abstract is encoded into a 384-dimensional dense vector (L2-normalised), and an SVM is trained over those vectors at query time to rank papers by relevance to your tags.

#### Email digests (optional)

To send periodic recommendation emails to users, see `send_emails.py`. You'll need to `uv add sendgrid` and configure your SendGrid credentials. Run this script in a daily cron job.

#### Todos

- Make website mobile friendly with media queries in css etc
- The metas table should not be a sqlitedict but a proper sqlite table, for efficiency
- Build a reverse index to support faster search, right now we iterate through the entire database

#### License

MIT
