#! /usr/bin/env python3

import settings
import pprint

token = settings.GITHUB_TOKEN
start_date = "2023-01-01T00:00:00Z"  # Start date (inclusive)
end_date = "2025-12-31T23:59:59Z"    # End date (inclusive)
commit_sample_percentage = 0.1  # 10% of PRs will be sampled for commit analysis
enable_outlier_rejection = True  # Set to False to disable outlier rejection
outlier_rejection_threshold = 2.0  # Number of standard deviations from the mean (e.g., 3.0 = remove PRs beyond 3 std devs)
repositories = settings.REPOSITORIES



import json
import os
import time
import pandas as pd
from datetime import datetime
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

# Set up GraphQL client
headers = {"Authorization": f"Bearer {token}"}
_transport = RequestsHTTPTransport(url="https://api.github.com/graphql", use_json=True, headers=headers)
client = Client(transport=_transport, fetch_schema_from_transport=True)

# ==============================================
# NEW: GraphQL query to list repositories by owner
# ==============================================
owner_repos_query = gql(
    """
    query getOwnerRepos($owner: String!, $cursor: String) {
      repositoryOwner(login: $owner) {
        repositories(first: 100, after: $cursor, orderBy: {field: NAME, direction: ASC}) {
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            name
            isFork
            isArchived
          }
        }
      }
    }
    """
)

# GraphQL query to count total merged pull requests (for progress tracking)
# Uses Search API with date filtering in the query string
pr_count_query = gql(
  """
    query getPRCount($query: String!) {
      search(query: $query, type: ISSUE, first: 1) {
        issueCount
      }
    }
  """)

# GraphQL query to fetch pull requests with statistics
# Fetches PRs in descending order by creation date, filters client-side
# Not ideal, but unfortunately the only way
pr_query = gql(
  """
    query getPullRequests($owner: String!, $name: String!, $cursor: String) {
      repository(owner: $owner, name: $name) {
        pullRequests(first: 100, after: $cursor, states: MERGED, orderBy: {field: CREATED_AT, direction: DESC}) {
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            number
            title
            state
            createdAt
            mergedAt
            closedAt
            additions
            deletions
            changedFiles
            author {
              login
            }
            commits {
              totalCount
            }
          }
        }
      }
    }
  """)

# GraphQL query to fetch additional commits for a PR (for PRs with more than 100 commits)
pr_commits_query = gql(
  """
    query getPRCommits($owner: String!, $name: String!, $prNumber: Int!, $cursor: String) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $prNumber) {
          commits(first: 100, after: $cursor) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              commit {
                oid
                additions
                deletions
                committedDate
              }
            }
          }
        }
      }
    }
  """)

def get_pr_count(owner, name):
    """Get the total count of merged pull requests for a repository, filtered by date range.
    Uses Search API"""
    try:
        # Build search query with date filtering
        # Format: "repo:owner/name is:pr is:merged created:YYYY-MM-DD..YYYY-MM-DD"
        query_parts = [f"repo:{owner}/{name}", "is:pr", "is:merged"]

        if start_date and end_date:
            # Both dates specified - use range
            start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            query_parts.append(f"created:{start_date_str}..{end_date_str}")
        elif start_date:
            # Only start date - use >=
            start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            query_parts.append(f"created:>={start_date_str}")
        elif end_date:
            # Only end date - use <=
            end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            query_parts.append(f"created:<={end_date_str}")

        search_query = " ".join(query_parts)
        variables = {"query": search_query}

        result = client.execute(pr_count_query, variable_values=variables)
        if not result.get('search'):
            return None
        return result['search']['issueCount']
    except Exception as e:
        print(f"  Error getting PR count: {e}")
        return None

def fetch_commits_for_pr(owner, name, pr_number, commit_cursor=None):
    """Fetch all commits for a specific PR with pagination."""
    all_commits = []
    cursor = commit_cursor

    while True:
        try:
            variables = {
                "owner": owner,
                "name": name,
                "prNumber": pr_number,
                "cursor": cursor
            }
            result = client.execute(pr_commits_query, variable_values=variables)

            if not result.get('repository') or not result['repository'].get('pullRequest'):
                break

            commits_data = result['repository']['pullRequest']['commits']
            commit_nodes = commits_data['nodes']

            if not commit_nodes:
                break

            for commit_node in commit_nodes:
                commit = commit_node['commit']
                all_commits.append({
                    'commit_sha': commit['oid'],
                    'additions': commit.get('additions', 0),
                    'deletions': commit.get('deletions', 0),
                    'committed_date': commit.get('committedDate'),
                })

            page_info = commits_data['pageInfo']
            if not page_info['hasNextPage']:
                break

            cursor = page_info['endCursor']
            time.sleep(0.2)  # Small delay for nested pagination

        except Exception as e:
            print(f"    Error fetching commits for PR #{pr_number}: {e}")
            break

    return all_commits

def fetch_prs_for_repo(owner, name, total_count=None):
    """Fetch pull requests for a repository with pagination (without commit details).
    Uses Repository Pull Requests API and filters client-side by date range.
    Stops fetching when PRs are outside the date range."""
    all_prs = []
    cursor = None
    page_count = 0
    stopped_early = False

    # Parse date filters for client-side filtering
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None

    if start_date and end_date:
        date_filter_str = f"created: {pd.to_datetime(start_date).strftime('%Y-%m-%d')} to {pd.to_datetime(end_date).strftime('%Y-%m-%d')}"
    elif start_date:
        date_filter_str = f"created >= {pd.to_datetime(start_date).strftime('%Y-%m-%d')}"
    elif end_date:
        date_filter_str = f"created <= {pd.to_datetime(end_date).strftime('%Y-%m-%d')}"
    else:
        date_filter_str = "no date filter"

    print(f"\nFetching pull requests for {owner}/{name}...")
    print(f"  Filter: Merged PRs, {date_filter_str}")

    while True:
        try:
            variables = {"owner": owner, "name": name, "cursor": cursor}
            result = client.execute(pr_query, variable_values=variables)

            if not result.get('repository'):
                print(f"  Warning: Repository {owner}/{name} not found")
                break

            pr_results = result['repository']['pullRequests']
            nodes = pr_results['nodes']

            if not nodes:
                break

            prs_in_page = 0
            for pr in nodes:
                if not pr:
                    continue

                # Get created_at for filtering
                created_at = pd.to_datetime(pr.get('createdAt'))

                # Since we're fetching in DESC order by created_at, if we hit a PR before start_date, we can stop
                if start_dt and created_at < start_dt:
                    stopped_early = True
                    break

                # Skip PRs after end_date
                if end_dt and created_at > end_dt:
                    continue

                date_field = pr.get('mergedAt') or pr.get('closedAt') or pr.get('createdAt')

                pr_data = {
                    'repository_owner': owner,
                    'repository_name': name,
                    'pr_number': pr['number'],
                    'pr_title': pr.get('title', ''),
                    'pr_state': pr.get('state', ''),
                    'created_at': pr.get('createdAt'),
                    'merged_at': pr.get('mergedAt'),
                    'closed_at': pr.get('closedAt'),
                    'date': date_field,
                    'additions': pr.get('additions', 0),
                    'deletions': pr.get('deletions', 0),
                    'changed_files': pr.get('changedFiles', 0),
                    'commit_count': pr.get('commits', {}).get('totalCount', 0) if pr.get('commits') else 0,
                    'author_login': pr.get('author', {}).get('login') if pr.get('author') else None,
                }
                pr_data['modifications'] = pr_data['additions'] + pr_data['deletions']
                all_prs.append(pr_data)
                prs_in_page += 1

            page_count += 1
            if total_count and total_count > 0:
                progress = (len(all_prs) / total_count) * 100 if total_count > 0 else 0
                print(f"  Page {page_count}: {prs_in_page} PRs (in range) | Total: {len(all_prs)}/{total_count} ({progress:.1f}%)")
            else:
                print(f"  Page {page_count}: {prs_in_page} PRs (in range) | Total: {len(all_prs)}")

            # Stop if we hit PRs before our start date
            if stopped_early:
                print(f"  Stopped early: reached PRs before start date")
                break

            page_info = pr_results['pageInfo']
            if not page_info['hasNextPage']:
                break

            cursor = page_info['endCursor']

            # Rate limiting: sleep between requests to avoid hitting rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error fetching PRs: {e}")
            break

    print(f"  Completed: {len(all_prs)} PRs fetched")
    return all_prs

# ==============================================
# helper to expand "*" into all repos
# ==============================================
def list_repos_for_owner(owner, include_forks=False, include_archived=False):
    """Return a list of repository names for a given owner.
    By default, skips forks and archived repositories."""
    repos = []
    cursor = None

    print(f"\nListing repositories for owner '{owner}' (include_forks={include_forks}, include_archived={include_archived})")

    while True:
        try:
            variables = {"owner": owner, "cursor": cursor}
            result = client.execute(owner_repos_query, variable_values=variables)

            if not result.get("repositoryOwner"):
                print(f"  Warning: owner '{owner}' not found or access denied")
                break

            repo_conn = result["repositoryOwner"]["repositories"]
            nodes = repo_conn["nodes"]

            for node in nodes:
                if not include_forks and node["isFork"]:
                    continue
                if not include_archived and node["isArchived"]:
                    continue
                repos.append(node["name"])

            page_info = repo_conn["pageInfo"]
            if not page_info["hasNextPage"]:
                break

            cursor = page_info["endCursor"]
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error listing repositories for owner '{owner}': {e}")
            break

    print(f"  Found {len(repos)} repositories for owner '{owner}'")
    return repos

# Collect pull requests for all repositories (without commit details)
all_prs_data = []

# Track actual repositories analyzed for summary (handles wildcard expansion)
analyzed_repos = set()

for repo in repositories:
    owner = repo['owner']
    name = repo['name']

    # ==============================================
    # NEW: wildcard expansion
    # ==============================================
    if name == "*":
        repo_names = list_repos_for_owner(owner)
        print(f"\n{'='*60}")
        print(f"Owner '{owner}' wildcard: analyzing {len(repo_names)} repositories")
        print(f"{'='*60}")

        for repo_name in repo_names:
            print(f"\n{'='*60}")
            print(f"Analyzing PRs for {owner}/{repo_name}")
            print(f"{'='*60}")

            total_pr_count = get_pr_count(owner, repo_name)
            if total_pr_count is None:
                print(f"  Skipping {owner}/{repo_name} - repository not found")
                continue

            print(f"  Total merged PRs: {total_pr_count:,}")
            prs = fetch_prs_for_repo(owner, repo_name, total_pr_count)
            all_prs_data.extend(prs)
            analyzed_repos.add((owner, repo_name))
    else:
        print(f"\n{'='*60}")
        print(f"Analyzing PRs for {owner}/{name}")
        print(f"{'='*60}")

        total_pr_count = get_pr_count(owner, name)
        if total_pr_count is None:
            print(f"  Skipping {owner}/{name} - repository not found")
            continue

        print(f"  Total merged PRs: {total_pr_count:,}")
        prs = fetch_prs_for_repo(owner, name, total_pr_count)
        all_prs_data.extend(prs)
        analyzed_repos.add((owner, name))

# Store PRs in memory as a DataFrame
prs_df = pd.DataFrame(all_prs_data)

# Convert date columns to datetime (filtering already done in fetch_prs_for_repo)
if not prs_df.empty:
    prs_df['date'] = pd.to_datetime(prs_df['date'])
    prs_df['created_at'] = pd.to_datetime(prs_df['created_at'])
    if 'merged_at' in prs_df.columns:
        prs_df['merged_at'] = pd.to_datetime(prs_df['merged_at'])
    if 'closed_at' in prs_df.columns:
        prs_df['closed_at'] = pd.to_datetime(prs_df['closed_at'])

    # Sort by date (newest first)
    prs_df = prs_df.sort_values('date', ascending=False).reset_index(drop=True)

    # Note: PRs are already filtered to MERGED state in the query, but keeping this for safety
    before_state_filter = len(prs_df)
    prs_df = prs_df[prs_df['pr_state'] == 'MERGED']
    closed_pr_count = before_state_filter - len(prs_df)

    print(f"\n{'='*60}")
    print(f"=== PR FINAL SUMMARY ===")
    print(f"{'='*60}")
    print(f"Total PRs collected: {len(all_prs_data):,}")
    if closed_pr_count > 0:
        print(f"Non-merged (closed) PRs filtered out: {closed_pr_count:,}")
    # UPDATED: use actual analyzed repos, not len(repositories)
    print(f"Repositories analyzed: {len(analyzed_repos)}")
    if len(prs_df) > 0:
        print(f"\nDate range: {prs_df['date'].min()} to {prs_df['date'].max()}")
        print(f"\nTotal additions: {prs_df['additions'].sum():,}")
        print(f"Total deletions: {prs_df['deletions'].sum():,}")
        print(f"Total modifications: {prs_df['modifications'].sum():,}")
        print(f"Total commit count: {prs_df['commit_count'].sum():,}")
        print(f"Total files changed: {prs_df['changed_files'].sum():,}")

        print(f"\n=== First 10 PRs ===")
        pprint.pprint(prs_df.head(10))

    print(f"\n✓ PR data collected and stored in 'prs_df' DataFrame")
else:
    print("No PRs were collected. Please check your repository list, date range, and token.")



# Sample PRs evenly across months for commit analysis
# This allows us to only fetch commits for a subset of PRs if desired
if not prs_df.empty and commit_sample_percentage > 0:
    import numpy as np

    # Create year-month column for grouping
    prs_df['year_month'] = prs_df['date'].dt.to_period('M')

    # Sample PRs evenly across each month
    sampled_prs_list = []

    for year_month, group in prs_df.groupby('year_month'):
        # Calculate how many PRs to sample from this month
        n_prs_in_month = len(group)
        n_to_sample = max(1, int(np.ceil(n_prs_in_month * commit_sample_percentage)))

        # Sample PRs from this month
        sampled = group.sample(n=min(n_to_sample, n_prs_in_month), random_state=42)
        sampled_prs_list.append(sampled)

    # Combine sampled PRs
    sampled_prs_df = pd.concat(sampled_prs_list, ignore_index=True)

    # Create a list of PR identifiers for commit fetching
    prs_for_commit_analysis = sampled_prs_df[['repository_owner', 'repository_name', 'pr_number']].to_dict('records')

    print(f"\n{'='*60}")
    print(f"=== PR SAMPLING FOR COMMIT ANALYSIS ===")
    print(f"{'='*60}")
    print(f"Sample percentage: {commit_sample_percentage * 100:.1f}%")
    print(f"Total PRs: {len(prs_df):,}")
    print(f"Sampled PRs: {len(sampled_prs_df):,}")

    # Show distribution across months
    monthly_counts = prs_df.groupby('year_month').size()
    monthly_sampled = sampled_prs_df.groupby('year_month').size()

    print(f"\n=== Sampling Distribution by Month ===")
    sampling_dist = pd.DataFrame({
        'Total PRs': monthly_counts,
        'Sampled PRs': monthly_sampled,
        'Sample %': (monthly_sampled / monthly_counts * 100).round(1)
    })
    pprint.pprint(sampling_dist)

    print(f"\n✓ Sampled PRs stored in 'sampled_prs_df' DataFrame")
    print(f"✓ PR list for commit analysis stored in 'prs_for_commit_analysis'")
    print(f"\nYou can now fetch commits for these PRs in the next cell.")
elif not prs_df.empty and commit_sample_percentage == 0:
    print(f"\n{'='*60}")
    print(f"=== PR SAMPLING FOR COMMIT ANALYSIS ===")
    print(f"{'='*60}")
    print(f"Sample percentage is set to 0.0 - skipping commit analysis.")
    print(f"Set commit_sample_percentage > 0 in the configuration cell to enable commit analysis.")
    sampled_prs_df = pd.DataFrame()
    prs_for_commit_analysis = []
else:
    print("No PRs available for sampling.")
    sampled_prs_df = pd.DataFrame()
    prs_for_commit_analysis = []



# Fetch commits for sampled PRs
# This cell fetches detailed commit information for the PRs selected in the sampling step
if 'prs_for_commit_analysis' in globals() and len(prs_for_commit_analysis) > 0:
    print(f"\n{'='*60}")
    print(f"=== FETCHING COMMITS FOR SAMPLED PRs ===")
    print(f"{'='*60}")
    print(f"Total PRs to analyze: {len(prs_for_commit_analysis):,}")

    all_pr_commits_data = []
    total_prs = len(prs_for_commit_analysis)

    for idx, pr_info in enumerate(prs_for_commit_analysis, 1):
        owner = pr_info['repository_owner']
        name = pr_info['repository_name']
        pr_number = pr_info['pr_number']

        print(f"  Processing PR {idx}/{total_prs}: {owner}/{name}#{pr_number}")

        # Fetch all commits for this PR
        commits = fetch_commits_for_pr(owner, name, pr_number)

        # Add PR metadata to each commit
        for commit_data in commits:
            commit_data['repository_owner'] = owner
            commit_data['repository_name'] = name
            commit_data['pr_number'] = pr_number
            all_pr_commits_data.append(commit_data)

        # Small delay to avoid rate limiting
        time.sleep(0.2)

    # Store PR commits in memory as a DataFrame
    pr_commits_df = pd.DataFrame(all_pr_commits_data) if all_pr_commits_data else pd.DataFrame()

    # Process PR commits DataFrame
    if not pr_commits_df.empty:
        pr_commits_df['committed_date'] = pd.to_datetime(pr_commits_df['committed_date'])
        pr_commits_df['modifications'] = pr_commits_df['additions'] + pr_commits_df['deletions']

        # Filter by date range if specified
        if start_date:
            start_dt = pd.to_datetime(start_date)
            pr_commits_df = pr_commits_df[pr_commits_df['committed_date'] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            pr_commits_df = pr_commits_df[pr_commits_df['committed_date'] <= end_dt]

        # Sort by committed date (newest first)
        pr_commits_df = pr_commits_df.sort_values('committed_date', ascending=False).reset_index(drop=True)

        # Filter out commits with 0 additions and 0 deletions
        before_zero_filter_commits = len(pr_commits_df)
        pr_commits_df = pr_commits_df[(pr_commits_df['additions'] > 0) | (pr_commits_df['deletions'] > 0)]
        zero_filtered_commits_count = before_zero_filter_commits - len(pr_commits_df)

        print(f"\n{'='*60}")
        print(f"=== COMMIT FETCHING SUMMARY ===")
        print(f"{'='*60}")
        print(f"Total commits collected: {before_zero_filter_commits:,}")
        if zero_filtered_commits_count > 0:
            print(f"Commits with 0 additions and 0 deletions filtered out: {zero_filtered_commits_count:,}")
        print(f"Final commit count: {len(pr_commits_df):,}")
        print(f"Unique PRs with commits: {pr_commits_df['pr_number'].nunique():,}")

        if len(pr_commits_df) > 0:
            print(f"\nDate range: {pr_commits_df['committed_date'].min()} to {pr_commits_df['committed_date'].max()}")
            print(f"\nTotal additions: {pr_commits_df['additions'].sum():,}")
            print(f"Total deletions: {pr_commits_df['deletions'].sum():,}")
            print(f"Total modifications: {pr_commits_df['modifications'].sum():,}")

            print(f"\n=== First 10 Commits ===")
            pprint.pprint(pr_commits_df.head(10))

        print(f"\n✓ PR commits data collected and stored in 'pr_commits_df' DataFrame")
    else:
        print("No commits were collected.")
        pr_commits_df = pd.DataFrame()
elif 'prs_for_commit_analysis' in globals() and len(prs_for_commit_analysis) == 0:
    print(f"\n{'='*60}")
    print(f"=== FETCHING COMMITS FOR SAMPLED PRs ===")
    print(f"{'='*60}")
    print("No PRs selected for commit analysis (prs_for_commit_analysis is empty).")
    print("This may happen if commit_sample_percentage is 0 or no PRs matched the criteria.")
    pr_commits_df = pd.DataFrame()
else:
    print(f"\n{'='*60}")
    print(f"=== FETCHING COMMITS FOR SAMPLED PRs ===")
    print(f"{'='*60}")
    print("prs_for_commit_analysis not found. Please run the sampling cell first.")
    pr_commits_df = pd.DataFrame()



# Additional Analysis and Visualizations

# ============================================================================
# PR Analysis and Visualizations (shown first)
# ============================================================================
if not prs_df.empty:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    # Set up plotting style
    plt.style.use('default')

    # ============================================================================
    # Outlier Rejection (if enabled)
    # ============================================================================
    if enable_outlier_rejection:
        print("\n" + "="*60)
        print("=== OUTLIER REJECTION ===")
        print("="*60)

        # Calculate z-scores for modifications (additions + deletions)
        modifications_mean = prs_df['modifications'].mean()
        modifications_std = prs_df['modifications'].std()

        # Calculate z-scores
        z_scores = np.abs((prs_df['modifications'] - modifications_mean) / modifications_std)

        # Identify outliers (PRs beyond the threshold standard deviations)
        outlier_mask = z_scores > outlier_rejection_threshold
        outliers_count = outlier_mask.sum()

        if outliers_count > 0:
            print(f"Outlier rejection threshold: {outlier_rejection_threshold} standard deviations")
            print(f"Mean modifications: {modifications_mean:,.2f}")
            print(f"Standard deviation: {modifications_std:,.2f}")
            print(f"PRs before outlier rejection: {len(prs_df):,}")
            print(f"Outliers detected: {outliers_count:,}")

            # Show some statistics about outliers
            outliers_df = prs_df[outlier_mask]
            print(f"\nOutlier PRs statistics:")
            print(f"  Min modifications: {outliers_df['modifications'].min():,}")
            print(f"  Max modifications: {outliers_df['modifications'].max():,}")
            print(f"  Mean modifications: {outliers_df['modifications'].mean():,.2f}")
            print(f"  Median modifications: {outliers_df['modifications'].median():,.2f}")

            # Store original count before removal
            original_count = len(prs_df)

            # Remove outliers
            prs_df = prs_df[~outlier_mask].copy()
            print(f"\nPRs after outlier rejection: {len(prs_df):,}")
            print(f"Removed {outliers_count:,} outlier PRs ({outliers_count/original_count*100:.2f}% of original data)")
        else:
            print(f"Outlier rejection threshold: {outlier_rejection_threshold} standard deviations")
            print(f"No outliers detected. All PRs within {outlier_rejection_threshold} standard deviations.")
            print(f"Total PRs: {len(prs_df):,}")

        print("="*60 + "\n")
    else:
        print("\nOutlier rejection is disabled.\n")

    # 1. PRs over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Monthly PRs over time, normalized per author
    prs_df['year_month'] = prs_df['date'].dt.to_period('M')

    # Calculate total PRs per month
    monthly_prs_total = prs_df.groupby('year_month').size()

    # Calculate number of unique authors per month
    monthly_pr_authors_count = prs_df.groupby('year_month')['author_login'].nunique()

    # Normalize: PRs per author per month
    monthly_prs_normalized = monthly_prs_total / monthly_pr_authors_count

    monthly_prs_dates = monthly_prs_normalized.index.to_timestamp()

    axes[0, 0].plot(monthly_prs_dates, monthly_prs_normalized.values, marker='o', markersize=4)
    axes[0, 0].set_title('Monthly PRs Over Time (Normalized per Author)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Average PRs per Author')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # Monthly median PR length over time (to avoid outliers)
    monthly_pr_median = prs_df.groupby('year_month')['modifications'].median()
    monthly_pr_mean = prs_df.groupby('year_month')['modifications'].mean()

    # Convert Period index to datetime for plotting
    monthly_pr_median_dates = monthly_pr_median.index.to_timestamp()
    monthly_pr_mean_dates = monthly_pr_mean.index.to_timestamp()

    # Plot median on left Y-axis
    axes[0, 1].plot(monthly_pr_median_dates, monthly_pr_median.values, label='Median PR Length', color='blue', marker='o', markersize=4, alpha=0.7)
    axes[0, 1].set_title('Monthly Median & Mean PR Length')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Median PR Length (Modifications)', color='blue')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].tick_params(axis='y', labelcolor='blue')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot mean on right Y-axis
    ax2_pr = axes[0, 1].twinx()
    ax2_pr.plot(monthly_pr_mean_dates, monthly_pr_mean.values, label='Mean PR Length', color='orange', marker='s', markersize=4, alpha=0.7, linestyle='--')
    ax2_pr.set_ylabel('Mean PR Length (Modifications)', color='orange')
    ax2_pr.tick_params(axis='y', labelcolor='orange')

    # Add legend
    lines1_pr, labels1_pr = axes[0, 1].get_legend_handles_labels()
    lines2_pr, labels2_pr = ax2_pr.get_legend_handles_labels()
    axes[0, 1].legend(lines1_pr + lines2_pr, labels1_pr + labels2_pr, loc='upper left')

    # PRs by repository
    pr_repo_counts = prs_df.groupby(['repository_owner', 'repository_name']).size()
    axes[1, 0].bar(range(len(pr_repo_counts)), pr_repo_counts.values)
    axes[1, 0].set_title('Total PRs by Repository')
    axes[1, 0].set_xlabel('Repository')
    axes[1, 0].set_ylabel('Number of PRs')
    axes[1, 0].set_xticks(range(len(pr_repo_counts)))
    axes[1, 0].set_xticklabels([f"{owner}/{name}" for owner, name in pr_repo_counts.index], rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Top PR contributors
    if 'author_login' in prs_df.columns:
        top_pr_contributors = prs_df[prs_df['author_login'].notna()].groupby('author_login').size().sort_values(ascending=False).head(10)
        if len(top_pr_contributors) > 0:
            axes[1, 1].barh(range(len(top_pr_contributors)), top_pr_contributors.values)
            axes[1, 1].set_title('Top 10 PR Contributors by PR Count')
            axes[1, 1].set_xlabel('Number of PRs')
            axes[1, 1].set_yticks(range(len(top_pr_contributors)))
            axes[1, 1].set_yticklabels(top_pr_contributors.index)
            axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()

    # 1b. Monthly median and mean files changed per PR over time
    monthly_files_median = prs_df.groupby('year_month')['changed_files'].median()
    monthly_files_mean = prs_df.groupby('year_month')['changed_files'].mean()

    # Convert Period index to datetime for plotting
    monthly_files_median_dates = monthly_files_median.index.to_timestamp()
    monthly_files_mean_dates = monthly_files_mean.index.to_timestamp()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot median on left Y-axis
    ax.plot(monthly_files_median_dates, monthly_files_median.values, label='Median Files Changed', color='blue', marker='o', markersize=4, alpha=0.7)
    ax.set_title('Monthly Median & Mean Files Changed per PR')
    ax.set_xlabel('Date')
    ax.set_ylabel('Median Files Changed per PR', color='blue')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.3)

    # Plot mean on right Y-axis
    ax2_files = ax.twinx()
    ax2_files.plot(monthly_files_mean_dates, monthly_files_mean.values, label='Mean Files Changed', color='orange', marker='s', markersize=4, alpha=0.7, linestyle='--')
    ax2_files.set_ylabel('Mean Files Changed per PR', color='orange')
    ax2_files.tick_params(axis='y', labelcolor='orange')

    # Add legend
    lines1_files, labels1_files = ax.get_legend_handles_labels()
    lines2_files, labels2_files = ax2_files.get_legend_handles_labels()
    ax.legend(lines1_files + lines2_files, labels1_files + labels2_files, loc='upper left')

    plt.tight_layout()
    plt.show()

    # Print statistics about files per PR
    print("\n=== Files per PR Statistics ===")
    print(f"Overall mean files per PR: {prs_df['changed_files'].mean():.2f}")
    print(f"Overall median files per PR: {prs_df['changed_files'].median():.2f}")
    print(f"Overall std dev files per PR: {prs_df['changed_files'].std():.2f}")
    print(f"Min files per PR: {prs_df['changed_files'].min()}")
    print(f"Max files per PR: {prs_df['changed_files'].max()}")

    # 2. Statistics by PR author
    if 'author_login' in prs_df.columns:
        print("\n=== Top PR Contributors ===")
        pr_author_stats = prs_df[prs_df['author_login'].notna()].groupby('author_login').agg({
            'pr_number': 'count',
            'additions': 'sum',
            'deletions': 'sum',
            'modifications': 'sum',
            'commit_count': 'sum',
            'changed_files': 'sum'
        }).sort_values('pr_number', ascending=False).head(10)
        pr_author_stats.columns = ['PRs', 'Additions', 'Deletions', 'Total Modifications', 'Total Commits', 'Total Files Changed']
        pprint.pprint(pr_author_stats)

    # 3. Monthly PR statistics - using median PR length to avoid outliers
    monthly_pr_stats = prs_df.groupby('year_month').agg({
        'pr_number': 'count',
        'modifications': ['median', 'mean'],
        'commit_count': ['median', 'mean', 'sum'],
        'changed_files': ['median', 'mean', 'sum']
    })
    monthly_pr_stats.columns = [
        'PRs',
        'Median_PR_Length', 'Mean_PR_Length',
        'Median_Commits_Per_PR', 'Mean_Commits_Per_PR', 'Total_Commits',
        'Median_Files_Per_PR', 'Mean_Files_Per_PR', 'Total_Files'
    ]

    # Add deviation from mean
    monthly_pr_stats['Deviation_From_Mean'] = monthly_pr_stats['Median_PR_Length'] - monthly_pr_stats['Mean_PR_Length']

    # ============================================================================
    # Commits per PR Analysis (if commit data is available)
    # ============================================================================
    # Check if pr_commits_df exists and has data
    has_commit_data = 'pr_commits_df' in globals() and not pr_commits_df.empty

    if has_commit_data:
        print("\n" + "="*60)
        print("=== COMMITS PER PR ANALYSIS ===")
        print("="*60)

        # Calculate commits per PR statistics
        commits_per_pr = pr_commits_df.groupby(['repository_owner', 'repository_name', 'pr_number']).agg({
            'commit_sha': 'count',
            'additions': 'sum',
            'deletions': 'sum',
            'modifications': ['mean', 'median']
        })
        commits_per_pr.columns = ['Commit_Count', 'Total_Additions', 'Total_Deletions', 'Mean_Commit_Size', 'Median_Commit_Size']
        commits_per_pr = commits_per_pr.reset_index()

        # Merge commit stats with PR data for monthly analysis
        pr_dates = prs_df[['repository_owner', 'repository_name', 'pr_number', 'date']].copy()
        commits_per_pr_with_date = commits_per_pr.merge(pr_dates, on=['repository_owner', 'repository_name', 'pr_number'], how='left')
        commits_per_pr_with_date['year_month'] = commits_per_pr_with_date['date'].dt.to_period('M')

        # Monthly statistics for commits per PR (from actual commit data)
        monthly_commits_per_pr_stats = commits_per_pr_with_date.groupby('year_month').agg({
            'Commit_Count': ['mean', 'median'],
            'Median_Commit_Size': ['mean', 'median'],
            'Mean_Commit_Size': ['mean', 'median']
        })
        monthly_commits_per_pr_stats.columns = [
            'Mean_Commits_Per_PR_Actual', 'Median_Commits_Per_PR_Actual',
            'Mean_Median_Commit_Size', 'Median_Median_Commit_Size',
            'Mean_Mean_Commit_Size', 'Median_Mean_Commit_Size'
        ]

        # Visualizations for commits per PR
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Distribution of commits per PR
        axes[0, 0].hist(commits_per_pr['Commit_Count'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution of Commits per PR')
        axes[0, 0].set_xlabel('Number of Commits per PR')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].axvline(commits_per_pr['Commit_Count'].median(), color='red', linestyle='--', label=f'Median: {commits_per_pr["Commit_Count"].median():.1f}')
        axes[0, 0].axvline(commits_per_pr['Commit_Count'].mean(), color='orange', linestyle='--', label=f'Mean: {commits_per_pr["Commit_Count"].mean():.1f}')
        axes[0, 0].legend()

        # 2. Median commit size per PR over time
        monthly_median_commit_size = commits_per_pr_with_date.groupby('year_month')['Median_Commit_Size'].median()
        monthly_mean_commit_size = commits_per_pr_with_date.groupby('year_month')['Mean_Commit_Size'].mean()

        monthly_median_commit_dates = monthly_median_commit_size.index.to_timestamp()
        monthly_mean_commit_dates = monthly_mean_commit_size.index.to_timestamp()

        axes[0, 1].plot(monthly_median_commit_dates, monthly_median_commit_size.values, label='Median Commit Size per PR', color='blue', marker='o', markersize=4, alpha=0.7)
        axes[0, 1].set_title('Monthly Median & Mean Commit Size per PR')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Median Commit Size (Modifications)', color='blue')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].tick_params(axis='y', labelcolor='blue')
        axes[0, 1].grid(True, alpha=0.3)

        ax2_commit = axes[0, 1].twinx()
        ax2_commit.plot(monthly_mean_commit_dates, monthly_mean_commit_size.values, label='Mean Commit Size per PR', color='orange', marker='s', markersize=4, alpha=0.7, linestyle='--')
        ax2_commit.set_ylabel('Mean Commit Size (Modifications)', color='orange')
        ax2_commit.tick_params(axis='y', labelcolor='orange')

        lines1_commit, labels1_commit = axes[0, 1].get_legend_handles_labels()
        lines2_commit, labels2_commit = ax2_commit.get_legend_handles_labels()
        axes[0, 1].legend(lines1_commit + lines2_commit, labels1_commit + labels2_commit, loc='upper left')

        # 3. Average commits per PR over time
        monthly_avg_commits_per_pr = commits_per_pr_with_date.groupby('year_month')['Commit_Count'].mean()
        monthly_avg_commits_dates = monthly_avg_commits_per_pr.index.to_timestamp()

        axes[1, 0].plot(monthly_avg_commits_dates, monthly_avg_commits_per_pr.values, marker='o', markersize=4, color='green')
        axes[1, 0].set_title('Average Commits per PR Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Average Commits per PR')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Scatter: PR size vs number of commits
        axes[1, 1].scatter(commits_per_pr['Commit_Count'], commits_per_pr['Total_Additions'] + commits_per_pr['Total_Deletions'],
                           alpha=0.5, s=20)
        axes[1, 1].set_title('PR Size vs Number of Commits')
        axes[1, 1].set_xlabel('Number of Commits per PR')
        axes[1, 1].set_ylabel('Total PR Size (Additions + Deletions)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Statistics summary
        print("\n=== Commits per PR Statistics ===")
        print(f"Mean commits per PR: {commits_per_pr['Commit_Count'].mean():.2f}")
        print(f"Median commits per PR: {commits_per_pr['Commit_Count'].median():.2f}")
        print(f"Std dev commits per PR: {commits_per_pr['Commit_Count'].std():.2f}")
        print(f"Min commits per PR: {commits_per_pr['Commit_Count'].min()}")
        print(f"Max commits per PR: {commits_per_pr['Commit_Count'].max()}")

        print("\n=== Commit Size per PR Statistics ===")
        print(f"Mean commit size (median per PR): {commits_per_pr['Median_Commit_Size'].mean():.2f} modifications")
        print(f"Median commit size (median per PR): {commits_per_pr['Median_Commit_Size'].median():.2f} modifications")
        print(f"Mean commit size (mean per PR): {commits_per_pr['Mean_Commit_Size'].mean():.2f} modifications")
        print(f"Median commit size (mean per PR): {commits_per_pr['Mean_Commit_Size'].median():.2f} modifications")

    # ============================================================================
    # Monthly Statistics Table
    # ============================================================================
    print("\n" + "="*60)
    print("=== MONTHLY PT STATISTICS ===")
    print("="*60)

    # Merge PR stats with commit stats if available
    if has_commit_data:
        # Merge the monthly stats
        unified_monthly_stats = monthly_pr_stats.merge(
            monthly_commits_per_pr_stats,
            left_index=True,
            right_index=True,
            how='left'
        )
        # Round numeric columns for better readability
        unified_monthly_stats = unified_monthly_stats.round(2)
    else:
        unified_monthly_stats = monthly_pr_stats.round(2)

    # Select and reorder columns for better presentation
    display_columns = [
        'PRs',
        'Median_PR_Length', 'Mean_PR_Length',
        'Median_Commits_Per_PR', 'Mean_Commits_Per_PR', 'Total_Commits',
        'Median_Files_Per_PR', 'Mean_Files_Per_PR', 'Total_Files'
    ]

    if has_commit_data:
        display_columns.extend([
            'Mean_Commits_Per_PR_Actual', 'Median_Commits_Per_PR_Actual',
            'Mean_Median_Commit_Size', 'Median_Median_Commit_Size'
        ])

    # Only include columns that exist
    display_columns = [col for col in display_columns if col in unified_monthly_stats.columns]

    pprint.pprint(unified_monthly_stats[display_columns])
    # save the dataframe to a csv
    unified_monthly_stats.to_csv('analysis.csv', index=False)

    if has_commit_data:
        print("\nNote: 'Commits_Per_PR' from PR metadata vs 'Commits_Per_PR_Actual' from commit analysis")
    else:
        print("\nNote: Commit analysis not available. Set commit_sample_percentage > 0 to enable commit analysis.")
else:
    print("No PR data available for analysis.")