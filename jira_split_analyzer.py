#!/usr/bin/env python3
"""
Jira Project Split Analyzer

Analyzes inter-project relationships in a Jira Cloud instance and recommends
the optimal way to split projects across two sites, minimizing cross-site links.

Relationships analyzed:
  - Issue links (blocks, relates to, duplicates, etc.)
  - Parent-child (epic-story, story-subtask)
  - Mentions (cross-project issue keys referenced in descriptions/comments)

Usage:
  pip install requests networkx python-louvain tabulate
  export JIRA_URL=https://yoursite.atlassian.net
  export JIRA_EMAIL=you@example.com
  export JIRA_API_TOKEN=your-api-token
  python jira_split_analyzer.py
"""

import os
import re
import sys
import json
import logging
from collections import defaultdict
from itertools import combinations

import requests
import networkx as nx
from community import community_louvain
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jira API helpers
# ---------------------------------------------------------------------------

class JiraClient:
    def __init__(self, base_url: str, email: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (email, api_token)
        self.session.headers.update({"Accept": "application/json"})

    def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/rest/api/3/{path}"
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}/rest/api/3/{path}"
        resp = self.session.post(url, json=body)
        resp.raise_for_status()
        return resp.json()

    def get_all_projects(self) -> list[dict]:
        """Return all projects the user can see."""
        projects = []
        start = 0
        while True:
            data = self.get("project/search", {"startAt": start, "maxResults": 50})
            projects.extend(data["values"])
            if data["isLast"]:
                break
            start += len(data["values"])
        return projects

    def search_issues(self, jql: str, fields: str, max_results: int = 100):
        """Generator that pages through JQL search results using POST endpoint."""
        start = 0
        while True:
            data = self.post("search/jql", {
                "jql": jql,
                "startAt": start,
                "maxResults": max_results,
                "fields": [f.strip() for f in fields.split(",")],
            })
            yield from data["issues"]
            if start + max_results >= data["total"]:
                break
            start += max_results


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_relationships(jira: JiraClient, project_keys: set[str]):
    """
    Scan all issues and collect cross-project relationships.

    Returns a dict: (projA, projB) -> count   (sorted tuple keys)
    """
    edge_weights = defaultdict(int)
    issue_count_by_project = defaultdict(int)

    # Build regex to detect cross-project mentions like "PROJ-123"
    mention_re = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")

    fields = "project,issuelinks,parent,subtasks,description,comment"
    total_issues = 0

    for proj_key in sorted(project_keys):
        log.info(f"Scanning project {proj_key} ...")
        jql = f"project = {proj_key} ORDER BY created ASC"

        for issue in jira.search_issues(jql, fields):
            total_issues += 1
            src_proj = issue["fields"]["project"]["key"]
            issue_count_by_project[src_proj] += 1

            if total_issues % 500 == 0:
                log.info(f"  ... processed {total_issues} issues so far")

            # --- 1. Issue links ---
            for link in issue["fields"].get("issuelinks") or []:
                linked_issue = link.get("outwardIssue") or link.get("inwardIssue")
                if linked_issue:
                    dst_proj = linked_issue["fields"]["project"]["key"]
                    if dst_proj != src_proj and dst_proj in project_keys:
                        key = tuple(sorted([src_proj, dst_proj]))
                        edge_weights[key] += 1

            # --- 2. Parent-child (epic/parent → child) ---
            parent = issue["fields"].get("parent")
            if parent:
                dst_proj = parent["fields"]["project"]["key"]
                if dst_proj != src_proj and dst_proj in project_keys:
                    key = tuple(sorted([src_proj, dst_proj]))
                    edge_weights[key] += 1

            for sub in issue["fields"].get("subtasks") or []:
                dst_proj = sub["fields"]["project"]["key"]
                if dst_proj != src_proj and dst_proj in project_keys:
                    key = tuple(sorted([src_proj, dst_proj]))
                    edge_weights[key] += 1

            # --- 3. Mentions in description ---
            desc = issue["fields"].get("description")
            if desc:
                # Jira Cloud v3 returns ADF; convert to plain text for scanning
                desc_text = _adf_to_text(desc) if isinstance(desc, dict) else str(desc)
                for match in mention_re.findall(desc_text):
                    dst_proj = match.rsplit("-", 1)[0]
                    if dst_proj != src_proj and dst_proj in project_keys:
                        key = tuple(sorted([src_proj, dst_proj]))
                        edge_weights[key] += 1

            # --- 4. Mentions in comments ---
            comment_data = issue["fields"].get("comment")
            if comment_data:
                for c in comment_data.get("comments") or []:
                    body = c.get("body", "")
                    body_text = _adf_to_text(body) if isinstance(body, dict) else str(body)
                    for match in mention_re.findall(body_text):
                        dst_proj = match.rsplit("-", 1)[0]
                        if dst_proj != src_proj and dst_proj in project_keys:
                            key = tuple(sorted([src_proj, dst_proj]))
                            edge_weights[key] += 1

    log.info(f"Finished scanning {total_issues} issues across {len(project_keys)} projects.")
    return dict(edge_weights), dict(issue_count_by_project)


def _adf_to_text(node) -> str:
    """Recursively extract plain text from Atlassian Document Format."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        if node.get("type") == "text":
            return node.get("text", "")
        return " ".join(_adf_to_text(child) for child in node.get("content", []))
    if isinstance(node, list):
        return " ".join(_adf_to_text(item) for item in node)
    return ""


# ---------------------------------------------------------------------------
# Graph analysis
# ---------------------------------------------------------------------------

def build_graph(project_keys: set[str], edge_weights: dict, issue_counts: dict) -> nx.Graph:
    G = nx.Graph()
    for pk in project_keys:
        G.add_node(pk, issues=issue_counts.get(pk, 0))
    for (a, b), w in edge_weights.items():
        G.add_edge(a, b, weight=w)
    return G


def analyze_communities(G: nx.Graph) -> dict[str, int]:
    """Louvain community detection — finds natural clusters."""
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    return partition


def find_best_bisection(G: nx.Graph) -> tuple[set[str], set[str], int]:
    """
    Try Kernighan-Lin bisection to find the min-cut split into two groups.
    Falls back to spectral bisection if KL fails.
    """
    try:
        partition = nx.community.kernighan_lin_bisection(G, weight="weight", seed=42)
        a, b = partition
    except Exception:
        # Fallback: spectral bisection via Fiedler vector
        fiedler = nx.fiedler_vector(G, weight="weight")
        nodes = list(G.nodes())
        a = {nodes[i] for i, v in enumerate(fiedler) if v < 0}
        b = {nodes[i] for i, v in enumerate(fiedler) if v >= 0}

    cut_weight = sum(
        G[u][v].get("weight", 1)
        for u, v in G.edges()
        if (u in a and v in b) or (u in b and v in a)
    )
    return a, b, cut_weight


def compute_cut_weight(G: nx.Graph, set_a: set[str], set_b: set[str]) -> int:
    return sum(
        G[u][v].get("weight", 1)
        for u, v in G.edges()
        if (u in set_a and v in set_b) or (u in set_b and v in set_a)
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    G: nx.Graph,
    edge_weights: dict,
    issue_counts: dict,
    communities: dict,
    site_a: set[str],
    site_b: set[str],
    cut_weight: int,
):
    total_links = sum(edge_weights.values())
    print("\n" + "=" * 70)
    print("JIRA PROJECT SPLIT ANALYSIS")
    print("=" * 70)

    # --- Summary ---
    print(f"\nProjects analyzed:   {len(G.nodes())}")
    print(f"Total issues:        {sum(issue_counts.values()):,}")
    print(f"Cross-project links: {total_links:,}")
    print(f"Project pairs with links: {len(edge_weights)}")

    # --- Top cross-project links ---
    print("\n--- Top 20 Cross-Project Relationships ---")
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)[:20]
    table = [[a, b, w] for (a, b), w in sorted_edges]
    print(tabulate(table, headers=["Project A", "Project B", "Links"], tablefmt="simple"))

    # --- Community detection ---
    num_communities = len(set(communities.values()))
    print(f"\n--- Natural Clusters (Louvain detected {num_communities} communities) ---")
    clusters = defaultdict(list)
    for proj, cid in communities.items():
        clusters[cid].append(proj)
    for cid, members in sorted(clusters.items()):
        issues = sum(issue_counts.get(p, 0) for p in members)
        print(f"  Cluster {cid}: {', '.join(sorted(members))}  ({issues:,} issues)")

    # --- Recommended bisection ---
    print(f"\n--- Recommended 2-Way Split (min-cut = {cut_weight:,} cross-site links) ---")
    issues_a = sum(issue_counts.get(p, 0) for p in site_a)
    issues_b = sum(issue_counts.get(p, 0) for p in site_b)
    print(f"\n  Site A ({len(site_a)} projects, {issues_a:,} issues):")
    for p in sorted(site_a):
        print(f"    - {p}  ({issue_counts.get(p, 0):,} issues)")
    print(f"\n  Site B ({len(site_b)} projects, {issues_b:,} issues):")
    for p in sorted(site_b):
        print(f"    - {p}  ({issue_counts.get(p, 0):,} issues)")

    # --- Cross-site links that would break ---
    print(f"\n--- Links That Would Cross Sites ({cut_weight:,} total) ---")
    broken = []
    for (a, b), w in sorted_edges:
        if (a in site_a and b in site_b) or (a in site_b and b in site_a):
            broken.append([a, b, w])
    if broken:
        print(tabulate(broken[:20], headers=["Project A", "Project B", "Links"], tablefmt="simple"))
    else:
        print("  None — clean split!")

    # --- Quality metrics ---
    internal_a = sum(w for (u, v), w in edge_weights.items() if u in site_a and v in site_a)
    internal_b = sum(w for (u, v), w in edge_weights.items() if u in site_b and v in site_b)
    print(f"\n--- Split Quality ---")
    print(f"  Internal links (Site A): {internal_a:,}")
    print(f"  Internal links (Site B): {internal_b:,}")
    print(f"  Cross-site links:        {cut_weight:,}")
    if total_links > 0:
        pct = (1 - cut_weight / total_links) * 100
        print(f"  Links preserved:         {pct:.1f}%")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_url = os.environ.get("JIRA_URL")
    email = os.environ.get("JIRA_EMAIL")
    token = os.environ.get("JIRA_API_TOKEN")

    if not all([base_url, email, token]):
        print("Set these environment variables:")
        print("  JIRA_URL=https://yoursite.atlassian.net")
        print("  JIRA_EMAIL=you@example.com")
        print("  JIRA_API_TOKEN=your-api-token")
        sys.exit(1)

    jira = JiraClient(base_url, email, token)

    # Discover projects
    log.info("Fetching projects ...")
    projects = jira.get_all_projects()
    project_keys = {p["key"] for p in projects}
    log.info(f"Found {len(project_keys)} projects: {', '.join(sorted(project_keys))}")

    if len(project_keys) < 2:
        print("Need at least 2 projects to analyze a split.")
        sys.exit(0)

    # Collect relationships
    edge_weights, issue_counts = collect_relationships(jira, project_keys)

    if not edge_weights:
        print("No cross-project relationships found. Any split works equally well.")
        sys.exit(0)

    # Build graph & analyze
    G = build_graph(project_keys, edge_weights, issue_counts)
    communities = analyze_communities(G)
    site_a, site_b, cut_weight = find_best_bisection(G)

    # Report
    print_report(G, edge_weights, issue_counts, communities, site_a, site_b, cut_weight)

    # Export raw data
    output = {
        "edge_weights": {f"{a}|{b}": w for (a, b), w in edge_weights.items()},
        "issue_counts": issue_counts,
        "communities": communities,
        "recommended_split": {
            "site_a": sorted(site_a),
            "site_b": sorted(site_b),
            "cut_weight": cut_weight,
        },
    }
    with open("jira_split_results.json", "w") as f:
        json.dump(output, f, indent=2)
    log.info("Raw data saved to jira_split_results.json")


if __name__ == "__main__":
    main()
