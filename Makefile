
# Update the database with newest papers
up:
	uv run python arxiv_daemon.py --num 2000
	uv run python compute.py

# Run the dev server
fun:
	uv run flask --app serve run
