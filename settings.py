import os
import dotenv
dotenv.load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
#these are 30 day tokens, you _will_ need to regenerate: https://github.com/settings/personal-access-tokens/new?contents=read&metadata=read&pull_requests=read


if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is not set")

# List of repositories to analyze
# Format: [{"owner": "owner_name", "name": "repo_name"}, ...]
# Note: This will fetch ALL merged PRs for each repo, not just ones on a specific branch
REPOSITORIES = [
    {"owner": "joram", "name": "john.oram.ca"},
    # Add more repositories here as needed
    # {"owner": "owner2", "name": "repo2"},
]
