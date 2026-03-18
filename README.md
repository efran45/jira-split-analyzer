# Jira Split Analyzer

Analyzes inter-project relationships in a Jira Cloud instance and recommends the optimal way to split projects across two sites, minimizing cross-site links.

## Relationships Analyzed

- **Issue links** — blocks, relates to, duplicates, clones, etc.
- **Parent-child** — epic→story, story→subtask across projects
- **Mentions** — issue keys referenced in descriptions and comments

## Output

1. **Top cross-project relationships** — which project pairs are most connected
2. **Natural clusters** — Louvain community detection shows organic groupings
3. **Recommended 2-way split** — Kernighan-Lin min-cut bisection that minimizes broken links
4. **Split quality metrics** — how many links are preserved vs. broken
5. **JSON export** — raw data saved to `jira_split_results.json`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
export JIRA_URL=https://yoursite.atlassian.net
export JIRA_EMAIL=you@example.com
export JIRA_API_TOKEN=your-api-token

python jira_split_analyzer.py
```

Generate an API token at https://id.atlassian.com/manage-profile/security/api-tokens.

## Notes

- On large instances (10k+ issues) this will take a while as it pages through every issue. Progress is logged.
- The Jira Cloud API has rate limits (~100 req/sec). The script uses reasonable page sizes.
- The algorithm optimizes for **minimum broken links**, not equal-sized sites.
