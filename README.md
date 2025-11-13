# GitHub PR Analysis Tool

A Python script that analyzes GitHub pull requests and generates statistics and visualizations. Because apparently you need to know how many lines of code you've been churning out, eh?

## What Does This Thing Do?

This tool fetches merged pull request data from GitHub repositories using the GraphQL API and generates a bunch of charts and statistics about your team's PR activity. It'll tell you:

- How many PRs are being merged over time (normalized per author)
- Average PR size (additions + deletions)
- Files changed per PR
- Top contributors by PR count
- Commit analysis (if you enable sampling)
- Monthly statistics broken down by repository

It also has outlier rejection so those massive "oops I committed node_modules" PRs don't screw up your stats.

## Prerequisites

- Python 3.x (obviously)
- A GitHub Personal Access Token with the following permissions:
  - `contents:read`
  - `metadata:read`
  - `pull_requests:read`
  
Get your token here: https://github.com/settings/personal-access-tokens/new?contents=read&metadata=read&pull_requests=read

## Installation & Setup

The lazy way (recommended):

```bash
./RUNME.sh
```

This script will:
1. Check for python3-venv (installs it if you're on Debian/Ubuntu)
2. Create a virtual environment
3. Install dependencies from `requirements.txt`
4. Prompt you for your GitHub token and save it to `.env`
5. Run the analysis script

The manual way (for when you're feeling ambitious):

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your token
echo "GITHUB_TOKEN=your_token_here" > .env

# Run the script
python3 analyis.py
```

## Configuration

Edit `settings.py` to configure which repositories to analyze:

```python
REPOSITORIES = [
    {"owner": "joram", "name": "john.oram.ca"},
    {"owner": "someorg", "name": "somerepo"},
    # Add more repositories here
]
```

### Wildcard Support

You can use `"*"` as the repository name to analyze ALL repositories for an owner:

```python
REPOSITORIES = [
    {"owner": "myorg", "name": "*"},  # Analyzes all repos under myorg
]
```

By default, this skips forks and archived repositories.

### Other Configuration Options

At the top of `analyis.py`, you can tweak:

- `start_date` and `end_date`: Filter PRs by creation date
- `commit_sample_percentage`: What percentage of PRs to sample for detailed commit analysis (default 10%)
  - Set to 0 to skip commit analysis entirely
- `enable_outlier_rejection`: Remove statistical outliers from your dataset (default True)
- `outlier_rejection_threshold`: Number of standard deviations for outlier detection (default 2.0)

## Output

The script generates:

1. **Console output**: Summary statistics about PRs, commits, and contributors
2. **Matplotlib charts**: 
   - Monthly PRs over time (normalized per author)
   - PR length trends (median & mean)
   - Files changed per PR over time
   - PRs by repository
   - Top contributors
   - Commit analysis charts (if enabled)
3. **CSV file**: `analysis.csv` containing monthly statistics

## Rate Limiting

The script includes delays between API calls to avoid hitting GitHub's rate limits:
- 0.5 seconds between PR pages
- 0.2 seconds between commit pages

If you're analyzing a ton of repositories, go make yourself a double-double. This might take a while.

## Notes

- GitHub Personal Access Tokens expire after 30 days, so you'll need to regenerate them. The link is in `settings.py` for your convenience.
- The script fetches PRs in descending order by creation date and stops early if it hits PRs outside your date range.
- Commit analysis is sampled evenly across months to avoid bias toward recent activity.
- There's a typo in the main script filename (`analyis.py` instead of `analysis.py`). Deal with it.

## Dependencies

- `gql[all]` - GraphQL client for GitHub API
- `pandas` - Data manipulation
- `matplotlib` - Charting
- `python-dotenv` - Environment variable management
- `neo4j-driver`, `gspread`, `oauth2client` - (Looks like these are leftover from some other project, but they're in requirements.txt so 🤷)

## Troubleshooting

**Script fails with authentication error**: 
- Check that your GitHub token is valid and has the correct permissions
- Make sure it's properly set in `.env`

**No PRs found**:
- Verify your repository names are correct
- Check your date range - maybe you don't have any PRs in that timeframe?
- Ensure the repositories actually have merged PRs

**Charts look weird**:
- Try adjusting the `outlier_rejection_threshold` 
- Check if you have enough data points (at least a few months of PRs)

## License

I don't see one, so... use at your own risk, I guess?

