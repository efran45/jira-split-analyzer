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

import argparse
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
        if not resp.ok:
            log.error(f"POST {path} returned {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
        return resp.json()

    def get_project_roles(self, project_key: str) -> dict:
        """Return {role_name: role_url} for a project."""
        return self.get(f"project/{project_key}/role")

    def get_role_actors(self, project_key: str, role_id: str) -> list[dict]:
        """Return the list of actors (users/groups) for a specific project role."""
        data = self.get(f"project/{project_key}/role/{role_id}")
        return data.get("actors", [])

    def get_all_projects(self) -> list[dict]:
        """
        Return all projects the user can see.

        Tries the paginated /project/search endpoint first (requires Browse
        Projects permission).  If that returns nothing, falls back to the older
        /project endpoint which is more permissive and returns a plain list.
        """
        projects = []
        start = 0
        while True:
            data = self.get("project/search", {"startAt": start, "maxResults": 50})
            projects.extend(data["values"])
            if data["isLast"]:
                break
            start += len(data["values"])

        if not projects:
            log.info("project/search returned 0 results — trying legacy /project endpoint …")
            result = self.get("project")
            # Legacy endpoint returns a plain list, not a paginated envelope
            if isinstance(result, list):
                projects = result

        return projects

    def search_issues(self, jql: str, fields: list[str], max_results: int = 100):
        """Generator that pages through JQL search results using POST endpoint."""
        next_page_token = None
        while True:
            body = {
                "jql": jql,
                "maxResults": max_results,
                "fields": fields,
            }
            if next_page_token:
                body["nextPageToken"] = next_page_token
            data = self.post("search/jql", body)
            yield from data["issues"]
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

    def get_issue_comments(self, issue_key: str) -> list[dict]:
        """Fetch comments for a single issue."""
        data = self.get(f"issue/{issue_key}/comment", {"maxResults": 100})
        return data.get("comments", [])


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _project_from_issue(issue_obj: dict) -> str | None:
    """Extract project key from an issue object.

    Handles both full objects (with fields.project.key) and minimal
    objects (with just a 'key' like 'PROJ-123').
    """
    try:
        return issue_obj["fields"]["project"]["key"]
    except (KeyError, TypeError):
        pass
    issue_key = issue_obj.get("key", "")
    if "-" in issue_key:
        return issue_key.rsplit("-", 1)[0]
    return None


CHECKPOINT_FILE = "jira_split_checkpoint.json"


def _load_checkpoint() -> dict:
    """Load checkpoint from disk if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        log.info(
            f"Loaded checkpoint: {len(data.get('completed_projects', []))} projects "
            f"already scanned ({data.get('total_issues', 0)} issues), "
            f"{len(data.get('roles_completed_projects', []))} projects with roles cached"
        )
        return data
    return {}


def _save_checkpoint(
    completed_projects: list[str],
    edge_weights: dict,
    issue_counts: dict,
    total_issues: int,
):
    """Save issue-relationship progress to disk, preserving any cached role data."""
    # Preserve user role fields written by _update_checkpoint_roles
    existing: dict = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            existing = json.load(f)

    data = {
        "completed_projects": completed_projects,
        "edge_weights": {f"{a}|{b}": w for (a, b), w in edge_weights.items()},
        "issue_counts": issue_counts,
        "total_issues": total_issues,
        "user_roles": existing.get("user_roles", {}),
        "roles_completed_projects": existing.get("roles_completed_projects", []),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _update_checkpoint_roles(proj_key: str, proj_data: dict, roles_completed: list[str]):
    """Merge user role data for one project into the checkpoint file."""
    existing: dict = {}
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            existing = json.load(f)

    existing.setdefault("user_roles", {})[proj_key] = {
        "users": sorted(proj_data["users"]),
        "groups": sorted(proj_data["groups"]),
        "roles": {
            role: {
                "users":  sorted(actors["users"]),
                "groups": sorted(actors["groups"]),
            }
            for role, actors in proj_data["roles"].items()
        },
    }
    existing["roles_completed_projects"] = roles_completed

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def collect_relationships(jira: JiraClient, project_keys: set[str], scan_comments: bool = False):
    """
    Scan all issues and collect cross-project relationships.
    Saves a checkpoint after each project so it can resume if interrupted.

    Returns a dict: (projA, projB) -> count   (sorted tuple keys)
    """
    edge_weights = defaultdict(int)
    issue_count_by_project = defaultdict(int)
    completed_projects = []

    # Load checkpoint if available, filtering to only the requested project_keys
    checkpoint = _load_checkpoint()
    if checkpoint:
        completed_projects = [
            p for p in checkpoint.get("completed_projects", []) if p in project_keys
        ]
        for k, w in checkpoint.get("edge_weights", {}).items():
            a, b = k.split("|")
            if a in project_keys and b in project_keys:
                edge_weights[(a, b)] = w
        issue_count_by_project.update({
            k: v for k, v in checkpoint.get("issue_counts", {}).items()
            if k in project_keys
        })

    # Build regex to detect cross-project mentions like "PROJ-123"
    mention_re = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")

    fields = ["project", "issuelinks", "parent", "subtasks", "description"]
    total_issues = checkpoint.get("total_issues", 0)
    remaining = sorted(project_keys - set(completed_projects))

    if remaining:
        log.info(f"{len(remaining)} projects remaining: {', '.join(remaining)}")
    else:
        log.info("All projects already scanned (from checkpoint).")

    for proj_key in remaining:
        log.info(f"Scanning project {proj_key} ...")
        jql = f'project = "{proj_key}" ORDER BY created ASC'
        proj_issues = 0

        for issue in jira.search_issues(jql, fields, max_results=5000):
            total_issues += 1
            proj_issues += 1
            issue_key = issue["key"]
            src_proj = issue["fields"]["project"]["key"]
            issue_count_by_project[src_proj] += 1

            if proj_issues % 1000 == 0:
                log.info(f"  ... {proj_issues} issues in {proj_key} ({total_issues} total)")

            # --- 1. Issue links ---
            for link in issue["fields"].get("issuelinks") or []:
                linked_issue = link.get("outwardIssue") or link.get("inwardIssue")
                if linked_issue:
                    dst_proj = _project_from_issue(linked_issue)
                    if dst_proj and dst_proj != src_proj and dst_proj in project_keys:
                        key = tuple(sorted([src_proj, dst_proj]))
                        edge_weights[key] += 1

            # --- 2. Parent-child (epic/parent → child) ---
            parent = issue["fields"].get("parent")
            if parent:
                dst_proj = _project_from_issue(parent)
                if dst_proj and dst_proj != src_proj and dst_proj in project_keys:
                    key = tuple(sorted([src_proj, dst_proj]))
                    edge_weights[key] += 1

            for sub in issue["fields"].get("subtasks") or []:
                dst_proj = _project_from_issue(sub)
                if dst_proj and dst_proj != src_proj and dst_proj in project_keys:
                    key = tuple(sorted([src_proj, dst_proj]))
                    edge_weights[key] += 1

            # --- 3. Mentions in description ---
            desc = issue["fields"].get("description")
            if desc:
                desc_text = _adf_to_text(desc) if isinstance(desc, dict) else str(desc)
                for match in mention_re.findall(desc_text):
                    dst_proj = match.rsplit("-", 1)[0]
                    if dst_proj != src_proj and dst_proj in project_keys:
                        key = tuple(sorted([src_proj, dst_proj]))
                        edge_weights[key] += 1

            # --- 4. Mentions in comments (opt-in, slow) ---
            if scan_comments:
                try:
                    comments = jira.get_issue_comments(issue_key)
                except Exception:
                    comments = []
                for c in comments:
                    body = c.get("body", "")
                    body_text = _adf_to_text(body) if isinstance(body, dict) else str(body)
                    for match in mention_re.findall(body_text):
                        dst_proj = match.rsplit("-", 1)[0]
                        if dst_proj != src_proj and dst_proj in project_keys:
                            key = tuple(sorted([src_proj, dst_proj]))
                            edge_weights[key] += 1

        # Save checkpoint after each project completes
        completed_projects.append(proj_key)
        _save_checkpoint(completed_projects, edge_weights, dict(issue_count_by_project), total_issues)
        log.info(f"  ✓ {proj_key} done ({proj_issues} issues). Checkpoint saved.")

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
# User / permission analysis
# ---------------------------------------------------------------------------

def collect_user_project_roles(jira: JiraClient, project_keys: set[str]) -> dict:
    """
    For every project, fetch all role actors (users and groups) via the Jira
    project-role API.  Results are checkpointed after each project so the
    fetch can resume if interrupted.

    Returns:
        {
            "PROJ_KEY": {
                "users":  {"accountId1", ...},   # all unique user accountIds
                "groups": {"groupName1", ...},   # all unique group names
                "roles":  {
                    "Developer": {"users": {...}, "groups": {...}},
                    ...
                }
            },
            ...
        }
    """
    result: dict = {}

    # Restore already-cached role data from the checkpoint
    checkpoint = _load_checkpoint()
    roles_completed: list[str] = checkpoint.get("roles_completed_projects", [])
    for proj_key, saved in checkpoint.get("user_roles", {}).items():
        if proj_key in project_keys:
            result[proj_key] = {
                "users":  set(saved["users"]),
                "groups": set(saved["groups"]),
                "roles": {
                    role: {
                        "users":  set(a["users"]),
                        "groups": set(a["groups"]),
                    }
                    for role, a in saved.get("roles", {}).items()
                },
            }

    remaining = sorted(project_keys - set(roles_completed))
    if roles_completed:
        log.info(
            f"Roles checkpoint: {len(roles_completed)} projects already cached, "
            f"{len(remaining)} remaining."
        )

    for proj_key in remaining:
        log.info(f"Fetching roles for {proj_key} …")
        proj_data: dict = {"users": set(), "groups": set(), "roles": {}}

        try:
            roles = jira.get_project_roles(proj_key)
        except Exception as exc:
            log.warning(f"  Could not fetch roles for {proj_key}: {exc}")
            result[proj_key] = proj_data
            roles_completed.append(proj_key)
            _update_checkpoint_roles(proj_key, proj_data, roles_completed)
            continue

        for role_name, role_url in roles.items():
            role_id = str(role_url).rstrip("/").split("/")[-1]
            try:
                actors = jira.get_role_actors(proj_key, role_id)
            except Exception as exc:
                log.warning(f"  Could not fetch actors for role '{role_name}' in {proj_key}: {exc}")
                continue

            role_users: set[str] = set()
            role_groups: set[str] = set()
            for actor in actors:
                if actor.get("type") == "atlassian-user-role-actor":
                    account_id = (actor.get("actorUser") or {}).get("accountId")
                    if account_id:
                        role_users.add(account_id)
                        proj_data["users"].add(account_id)
                elif actor.get("type") == "atlassian-group-role-actor":
                    group_name = (actor.get("actorGroup") or {}).get("name") or actor.get("displayName")
                    if group_name:
                        role_groups.add(group_name)
                        proj_data["groups"].add(group_name)

            proj_data["roles"][role_name] = {"users": role_users, "groups": role_groups}

        result[proj_key] = proj_data
        roles_completed.append(proj_key)
        _update_checkpoint_roles(proj_key, proj_data, roles_completed)
        log.info(
            f"  ✓ {proj_key}: {len(proj_data['users'])} users, "
            f"{len(proj_data['groups'])} groups across {len(roles)} roles. Checkpoint saved."
        )

    return result


def build_user_overlap_edges(user_data: dict) -> dict:
    """
    Build edge weights from shared users/groups between every pair of projects.
    Groups count 3× more than individual users (they represent whole teams).

    Returns: {(projA, projB): overlap_score}  (same tuple-key format as edge_weights)
    """
    overlap: dict = defaultdict(int)
    project_keys = list(user_data.keys())
    for a, b in combinations(project_keys, 2):
        shared_users = len(user_data[a]["users"] & user_data[b]["users"])
        shared_groups = len(user_data[a]["groups"] & user_data[b]["groups"])
        score = shared_users + shared_groups * 3
        if score > 0:
            overlap[tuple(sorted([a, b]))] += score
    return dict(overlap)


def analyze_user_split_impact(
    site_a: set[str],
    site_b: set[str],
    user_data: dict,
) -> dict:
    """
    Classify every user and group as belonging to Site A only, Site B only,
    or spanning both sites (requires provisioning on both).

    Returns a dict with six sets keyed by descriptive names.
    """
    users_a: set[str] = set()
    users_b: set[str] = set()
    groups_a: set[str] = set()
    groups_b: set[str] = set()

    for proj in site_a:
        if proj in user_data:
            users_a |= user_data[proj]["users"]
            groups_a |= user_data[proj]["groups"]
    for proj in site_b:
        if proj in user_data:
            users_b |= user_data[proj]["users"]
            groups_b |= user_data[proj]["groups"]

    return {
        "users_site_a_only":  users_a - users_b,
        "users_site_b_only":  users_b - users_a,
        "users_on_both":      users_a & users_b,
        "groups_site_a_only": groups_a - groups_b,
        "groups_site_b_only": groups_b - groups_a,
        "groups_on_both":     groups_a & groups_b,
    }


def print_user_report(
    impact: dict,
    user_data: dict,
    site_a: set[str],
    site_b: set[str],
):
    total_users = (
        len(impact["users_site_a_only"])
        + len(impact["users_site_b_only"])
        + len(impact["users_on_both"])
    )
    total_groups = (
        len(impact["groups_site_a_only"])
        + len(impact["groups_site_b_only"])
        + len(impact["groups_on_both"])
    )

    print("\n--- User & Permission Impact ---")
    print(f"  Total unique users:  {total_users}")
    print(f"  Total unique groups: {total_groups}")
    print()
    print(f"  Users exclusive to Site A:  {len(impact['users_site_a_only'])}")
    print(f"  Users exclusive to Site B:  {len(impact['users_site_b_only'])}")
    print(f"  Users spanning BOTH sites:  {len(impact['users_on_both'])}"
          + ("  ← need accounts on both sites" if impact["users_on_both"] else ""))
    print()
    print(f"  Groups exclusive to Site A: {len(impact['groups_site_a_only'])}")
    print(f"  Groups exclusive to Site B: {len(impact['groups_site_b_only'])}")
    print(f"  Groups spanning BOTH sites: {len(impact['groups_on_both'])}"
          + ("  ← must be duplicated" if impact["groups_on_both"] else ""))

    if impact["groups_on_both"]:
        print("\n  Groups to duplicate across both sites:")
        for g in sorted(impact["groups_on_both"]):
            print(f"    - {g}")

    if impact["groups_site_a_only"]:
        print("\n  Groups staying on Site A only:")
        for g in sorted(impact["groups_site_a_only"]):
            print(f"    - {g}")

    if impact["groups_site_b_only"]:
        print("\n  Groups staying on Site B only:")
        for g in sorted(impact["groups_site_b_only"]):
            print(f"    - {g}")

    # Per-project breakdown table
    print("\n  Per-project user/group counts:")
    table = []
    for proj in sorted(site_a | site_b):
        if proj in user_data:
            site_label = "A" if proj in site_a else "B"
            table.append([
                proj,
                site_label,
                len(user_data[proj]["users"]),
                len(user_data[proj]["groups"]),
                ", ".join(sorted(user_data[proj]["roles"].keys())),
            ])
    print(tabulate(table, headers=["Project", "Site", "Users", "Groups", "Roles"], tablefmt="simple"))


# ---------------------------------------------------------------------------
# Graph analysis
# ---------------------------------------------------------------------------

def build_graph(project_keys: set[str], edge_weights: dict, issue_counts: dict) -> nx.Graph:
    G = nx.Graph()
    for pk in project_keys:
        G.add_node(pk, issues=issue_counts.get(pk, 0))
    for (a, b), w in edge_weights.items():
        if a in project_keys and b in project_keys:
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
    parser = argparse.ArgumentParser(description="Analyze Jira projects for optimal site split")
    parser.add_argument(
        "--scan-comments", action="store_true",
        help="Also scan issue comments for cross-project mentions (much slower)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore checkpoint and start a fresh scan",
    )
    parser.add_argument(
        "--user-weight", type=float, default=10.0, metavar="W",
        help=(
            "Scale factor applied to user/group overlap edges before merging "
            "with issue-link edges (default: 10). Raise to make user affinity "
            "influence the split more; set to 0 to disable user weighting even "
            "when --include-users is active."
        ),
    )
    parser.add_argument(
        "--include-users", action="store_true",
        help="Fetch project role members and factor user/group overlap into the split",
    )
    args = parser.parse_args()

    if args.fresh and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log.info("Removed checkpoint file — starting fresh.")

    # Load saved defaults from .env if present
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    saved = {}
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    saved[k.strip()] = v.strip()

    # Always prompt, showing saved defaults
    def prompt(label, key, hide=False):
        default = saved.get(key, "")
        if default:
            display = f"{label} [{default}]: "
        else:
            display = f"{label}: "
        value = input(display).strip()
        return value if value else default

    print("\n— Jira Connection —\n")
    base_url = prompt("Jira URL (e.g. https://yoursite.atlassian.net)", "JIRA_URL")
    email = prompt("Jira email", "JIRA_EMAIL")
    token = prompt("Jira API token", "JIRA_API_TOKEN")

    if not all([base_url, email, token]):
        print("All three values are required.")
        sys.exit(1)

    # Save for next time if changed or no .env yet
    new_vals = {"JIRA_URL": base_url, "JIRA_EMAIL": email, "JIRA_API_TOKEN": token}
    if new_vals != saved:
        save = input("\nSave credentials for next time? (y/n): ").strip().lower()
        if save == "y":
            with open(env_file, "w") as f:
                f.write(f"JIRA_URL={base_url}\n")
                f.write(f"JIRA_EMAIL={email}\n")
                f.write(f"JIRA_API_TOKEN={token}\n")
            print(f"Saved to {env_file}")
    print()

    jira = JiraClient(base_url, email, token)

    # Verify connectivity and show which account is being used
    try:
        myself = jira.get("myself")
        log.info(
            f"Authenticated as: {myself.get('displayName')} "
            f"<{myself.get('emailAddress')}> (accountId: {myself.get('accountId')})"
        )
    except Exception as exc:
        print(f"\nAuthentication failed: {exc}")
        print("Check your Jira URL, email, and API token.")
        sys.exit(1)

    # Discover projects
    log.info("Fetching projects ...")
    projects = jira.get_all_projects()
    project_keys = {p["key"] for p in projects}
    log.info(f"Found {len(project_keys)} projects: {', '.join(sorted(project_keys))}")

    if len(project_keys) < 2:
        print("Need at least 2 projects to analyze a split.")
        sys.exit(0)

    # Decide whether to include user/permission analysis
    include_users = args.include_users
    if not include_users:
        ans = input("Include user/permission analysis? (y/n) [y]: ").strip().lower()
        include_users = ans != "n"

    # Collect issue-link relationships
    edge_weights, issue_counts = collect_relationships(jira, project_keys, scan_comments=args.scan_comments)

    if not edge_weights:
        print("No cross-project relationships found. Any split works equally well.")
        sys.exit(0)

    # Optionally collect user/group data and merge into edge weights
    user_data: dict = {}
    user_overlap_edges: dict = {}
    if include_users:
        log.info("Collecting user and group role assignments …")
        user_data = collect_user_project_roles(jira, project_keys)
        user_overlap_edges = build_user_overlap_edges(user_data)
        if user_overlap_edges and args.user_weight > 0:
            log.info(
                f"Merging {len(user_overlap_edges)} user-overlap edges "
                f"(weight ×{args.user_weight}) into the graph …"
            )
            merged: dict = dict(edge_weights)
            for key, score in user_overlap_edges.items():
                merged[key] = merged.get(key, 0) + score * args.user_weight
            edge_weights = merged
        else:
            log.info("No user/group overlap found between projects.")

    # Build graph & analyze
    G = build_graph(project_keys, edge_weights, issue_counts)
    communities = analyze_communities(G)
    site_a, site_b, cut_weight = find_best_bisection(G)

    # Report
    print_report(G, edge_weights, issue_counts, communities, site_a, site_b, cut_weight)

    # User impact report (appended to main report)
    user_impact: dict = {}
    if include_users and user_data:
        user_impact = analyze_user_split_impact(site_a, site_b, user_data)
        print_user_report(user_impact, user_data, site_a, site_b)
        print("\n" + "=" * 70)

    # Export raw data
    def _set_to_list(obj):
        """JSON-serialise sets as sorted lists."""
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    output: dict = {
        "edge_weights": {f"{a}|{b}": w for (a, b), w in edge_weights.items()},
        "issue_counts": issue_counts,
        "communities": communities,
        "recommended_split": {
            "site_a": sorted(site_a),
            "site_b": sorted(site_b),
            "cut_weight": cut_weight,
        },
    }
    if user_impact:
        output["user_permission_impact"] = {
            "users_site_a_only":  sorted(user_impact["users_site_a_only"]),
            "users_site_b_only":  sorted(user_impact["users_site_b_only"]),
            "users_on_both":      sorted(user_impact["users_on_both"]),
            "groups_site_a_only": sorted(user_impact["groups_site_a_only"]),
            "groups_site_b_only": sorted(user_impact["groups_site_b_only"]),
            "groups_on_both":     sorted(user_impact["groups_on_both"]),
        }
    if user_data:
        output["project_roles"] = {
            proj: {
                "users":  sorted(data["users"]),
                "groups": sorted(data["groups"]),
                "roles": {
                    role: {
                        "users":  sorted(actors["users"]),
                        "groups": sorted(actors["groups"]),
                    }
                    for role, actors in data["roles"].items()
                },
            }
            for proj, data in user_data.items()
        }

    with open("jira_split_results.json", "w") as f:
        json.dump(output, f, indent=2)
    log.info("Raw data saved to jira_split_results.json")


if __name__ == "__main__":
    main()
