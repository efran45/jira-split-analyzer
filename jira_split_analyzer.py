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


def build_category_affinity_edges(
    project_categories: dict,
    project_keys: set[str],
    weight: float = 100.0,
) -> dict:
    """
    Add affinity edges between every pair of projects that share the same
    project category.  A high weight strongly encourages the partitioning
    algorithms to keep same-category projects on the same site.

    Returns: {(projA, projB): weight}
    """
    edges: dict = defaultdict(float)
    by_category: dict = defaultdict(list)
    for proj, cat in project_categories.items():
        if proj in project_keys and cat:
            by_category[cat].append(proj)
    for members in by_category.values():
        if len(members) > 1:
            for a, b in combinations(sorted(members), 2):
                edges[tuple(sorted([a, b]))] += weight
    return dict(edges)


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


def analyze_user_disruption_multisite(sites: list[set], user_data: dict) -> dict:
    """
    Generalised user-impact analysis for any number of sites.

    The primary metric is the number of users who need to log into more
    than one site — the lower this number, the better the split.

    Returns:
        {
            "labels":          ["A", "B", ...],
            "sites":           [set_a, set_b, ...],
            "site_users":      [users_on_a, users_on_b, ...],
            "site_groups":     [groups_on_a, groups_on_b, ...],
            "spanning_users":  set of users who appear on 2+ sites,
            "spanning_groups": set of groups that exist on 2+ sites,
            "disruption_score": len(spanning_users),
        }
    """
    labels = [chr(65 + i) for i in range(len(sites))]  # A, B, C, …
    site_users: list[set] = []
    site_groups: list[set] = []
    for site in sites:
        users: set = set()
        groups: set = set()
        for proj in site:
            if proj in user_data:
                users  |= user_data[proj]["users"]
                groups |= user_data[proj]["groups"]
        site_users.append(users)
        site_groups.append(groups)

    all_users  = set().union(*site_users)  if site_users  else set()
    all_groups = set().union(*site_groups) if site_groups else set()

    spanning_users  = {u for u in all_users  if sum(1 for s in site_users  if u in s) > 1}
    spanning_groups = {g for g in all_groups if sum(1 for s in site_groups if g in s) > 1}

    return {
        "labels":           labels,
        "sites":            sites,
        "site_users":       site_users,
        "site_groups":      site_groups,
        "spanning_users":   spanning_users,
        "spanning_groups":  spanning_groups,
        "disruption_score": len(spanning_users),
    }


def user_disruption_score(sites: list[set], user_data: dict) -> int:
    """Quick helper — returns just the disruption score (users needing 2+ logins)."""
    if not user_data:
        return 0
    return analyze_user_disruption_multisite(sites, user_data)["disruption_score"]


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


def compute_multiway_cut(G: nx.Graph, groups: list[set]) -> float:
    """Total edge weight crossing any boundary in a k-way partition."""
    total = 0.0
    for u, v in G.edges():
        gu = next((i for i, g in enumerate(groups) if u in g), -1)
        gv = next((i for i, g in enumerate(groups) if v in g), -1)
        if gu != gv:
            total += G[u][v].get("weight", 1)
    return total


def find_best_tripartition(G: nx.Graph) -> tuple[set, set, set, float]:
    """
    Find a 3-way split via recursive KL bisection.

    Runs the initial bisection once (A vs BC), then tries splitting each
    half further and keeps whichever 3-partition has the lower multiway
    cut weight.

    Returns (site_a, site_b, site_c, cut_weight).
    Raises ValueError if the graph has fewer than 3 nodes.
    """
    if len(G.nodes()) < 3:
        raise ValueError("Need at least 3 projects for a 3-way split.")

    first, second, _ = find_best_bisection(G)
    best_groups: list[set] | None = None
    best_cut = float("inf")

    for keep, to_split in [(first, second), (second, first)]:
        if len(to_split) < 2:
            continue
        sub = G.subgraph(to_split).copy()
        try:
            x, y, _ = find_best_bisection(sub)
        except Exception:
            continue
        groups = [keep, x, y]
        cut = compute_multiway_cut(G, groups)
        if cut < best_cut:
            best_cut = cut
            best_groups = groups

    if best_groups is None:
        return first, second, set(), compute_multiway_cut(G, [first, second])

    return best_groups[0], best_groups[1], best_groups[2], best_cut


# ---------------------------------------------------------------------------
# Multi-objective scoring and optimisation
# ---------------------------------------------------------------------------

def score_partition(
    sites: list[set],
    user_data: dict,
    project_categories: dict,
    raw_edge_weights: dict,
) -> dict:
    """
    Evaluate a partition against all three objectives in priority order:

      1. User disruption  — users who must log into 2+ sites  (highest priority)
      2. Categories split — categories whose projects span multiple sites
      3. Cross-site links — sum of raw issue-link weights crossing sites

    Returns a dict with each metric plus a composite score (lower = better).
    The composite uses lexicographic-style weighting so objective 1 always
    dominates objective 2, which always dominates objective 3.
    """
    n = len(sites)

    def site_idx(proj: str) -> int:
        for i, s in enumerate(sites):
            if proj in s:
                return i
        return -1

    # 1. User disruption
    ud = user_disruption_score(sites, user_data)

    # 2. Categories split across sites
    cat_site_sets: dict = defaultdict(set)
    for i, site in enumerate(sites):
        for proj in site:
            cat = project_categories.get(proj, "")
            if cat:
                cat_site_sets[cat].add(i)
    cats_split = sum(1 for s in cat_site_sets.values() if len(s) > 1)

    # 3. Raw cross-site issue links
    cross_links = int(sum(
        w for (a, b), w in raw_edge_weights.items()
        if site_idx(a) != site_idx(b) and site_idx(a) >= 0 and site_idx(b) >= 0
    ))

    # Composite — objective 1 always overrides 2, objective 2 always overrides 3
    max_links = sum(raw_edge_weights.values()) or 1
    composite = (
        ud        * 1_000_000
        + cats_split * 1_000
        + (cross_links / max_links) * 100
    )

    return {
        "user_disruption":  ud,
        "categories_split": cats_split,
        "cross_site_links": cross_links,
        "composite":        composite,
        "n_sites":          n,
    }


def _category_first_partition(
    project_keys: set[str],
    project_categories: dict,
    user_data: dict,
    n_sites: int,
) -> list[set]:
    """
    Greedy category-first bin packing.

    Treats each project category as an atomic unit (never split across sites).
    Uncategorised projects are treated as individual units.
    Each unit is assigned to whichever current site minimises user disruption,
    with project-count balance as a tie-breaker.
    """
    by_cat: dict = defaultdict(set)
    uncategorised: set = set()
    for proj in project_keys:
        cat = project_categories.get(proj, "")
        if cat:
            by_cat[cat].add(proj)
        else:
            uncategorised.add(proj)

    units: list[set] = list(by_cat.values()) + [{p} for p in sorted(uncategorised)]
    units.sort(key=len, reverse=True)   # largest first for better packing

    sites: list[set] = [set() for _ in range(n_sites)]
    for unit in units:
        best_i, best_val = 0, float("inf")
        for i in range(n_sites):
            trial = [set(s) for s in sites]
            trial[i] |= unit
            ud = user_disruption_score(trial, user_data)
            balance = max(len(s) for s in trial) - min(len(s) for s in trial)
            val = ud * 10_000 + balance
            if val < best_val:
                best_val, best_i = val, i
        sites[best_i] |= unit

    return [s for s in sites if s]   # drop any empty sites


def local_search_improve(
    initial_sites: list[set],
    score_fn,
    max_moves: int = 500,
) -> tuple[list[set], dict]:
    """
    Greedy local search: repeatedly move one project to a different site if
    it strictly improves the composite score.  Stops when no improving move
    exists or max_moves is reached.
    """
    sites = [set(s) for s in initial_sites]
    current = score_fn(sites)
    moves = 0

    while moves < max_moves:
        best_delta_sites = None
        best_delta_score = None

        for from_i, site in enumerate(sites):
            for proj in list(site):
                for to_i in range(len(sites)):
                    if to_i == from_i:
                        continue
                    candidate = [set(s) for s in sites]
                    candidate[from_i].discard(proj)
                    if not candidate[from_i]:   # never empty a site
                        continue
                    candidate[to_i].add(proj)
                    sc = score_fn(candidate)
                    if sc["composite"] < current["composite"]:
                        if best_delta_score is None or sc["composite"] < best_delta_score["composite"]:
                            best_delta_sites = candidate
                            best_delta_score = sc

        if best_delta_sites is None:
            break   # local optimum

        sites = best_delta_sites
        current = best_delta_score
        moves += 1

    return sites, current


def find_optimal_split(
    G: nx.Graph,
    user_data: dict,
    project_categories: dict,
    raw_edge_weights: dict,
    max_sites: int = 3,
) -> tuple[list[set], dict, str]:
    """
    Try every viable initial partitioning strategy, apply local search to
    each, and return the partition that best satisfies — in strict priority
    order — user disruption, category cohesion, and cross-site link count.

    Returns (sites, score_dict, winning_strategy_name).
    """
    project_keys = set(G.nodes())

    def score_fn(s):
        return score_partition(s, user_data, project_categories, raw_edge_weights)

    candidates: list[tuple[list[set], str]] = []

    # ── 2-site strategies ────────────────────────────────────────────────────
    try:
        a, b, _ = find_best_bisection(G)
        candidates.append(([a, b], "KL bisection (2-site)"))
    except Exception:
        pass

    try:
        comm = analyze_communities(G)
        clusters: dict = defaultdict(set)
        for proj, cid in comm.items():
            clusters[cid].add(proj)
        sc = sorted(clusters.values(), key=len, reverse=True)
        if len(sc) >= 2:
            candidates.append(([sc[0], set().union(*sc[1:])], "Louvain → 2 sites"))
    except Exception:
        pass

    try:
        cp = _category_first_partition(project_keys, project_categories, user_data, 2)
        if len(cp) == 2:
            candidates.append((cp, "Category-first (2-site)"))
    except Exception:
        pass

    # ── 3-site strategies ────────────────────────────────────────────────────
    if max_sites >= 3 and len(project_keys) >= 3:
        try:
            a, b, c, _ = find_best_tripartition(G)
            if c:
                candidates.append(([a, b, c], "KL bisection (3-site)"))
        except Exception:
            pass

        try:
            comm = analyze_communities(G)
            clusters = defaultdict(set)
            for proj, cid in comm.items():
                clusters[cid].add(proj)
            sc = sorted(clusters.values(), key=len, reverse=True)
            if len(sc) >= 3:
                candidates.append((
                    [sc[0], sc[1], set().union(*sc[2:])],
                    "Louvain natural clusters (3-site)",
                ))
        except Exception:
            pass

        try:
            cp3 = _category_first_partition(project_keys, project_categories, user_data, 3)
            if len(cp3) == 3:
                candidates.append((cp3, "Category-first (3-site)"))
        except Exception:
            pass

    if not candidates:
        a, b, _ = find_best_bisection(G)
        return [a, b], score_fn([a, b]), "KL bisection (fallback)"

    # Apply local search to every candidate and keep the best result
    best_sites: list[set] | None = None
    best_score: dict | None = None
    best_name = ""

    for initial, name in candidates:
        improved, sc = local_search_improve(initial, score_fn)
        log.info(
            f"  [{name}] disruption={sc['user_disruption']}  "
            f"cats_split={sc['categories_split']}  "
            f"cross_links={sc['cross_site_links']:,}  "
            f"composite={sc['composite']:.1f}"
        )
        if best_score is None or sc["composite"] < best_score["composite"]:
            best_sites  = improved
            best_score  = sc
            best_name   = name

    return best_sites, best_score, best_name   # type: ignore[return-value]


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
    parser.add_argument(
        "--category-weight", type=float, default=100.0, metavar="W",
        help=(
            "Edge weight added between every pair of projects in the same category "
            "(default: 100). Higher values more strongly keep categories together."
        ),
    )
    parser.add_argument(
        "--max-sites", type=int, default=3, choices=[2, 3],
        help="Maximum number of sites to consider (default: 3). "
             "The tool still recommends 2 unless 3 clearly reduces user disruption.",
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

    # Discover projects and their categories
    log.info("Fetching projects ...")
    projects = jira.get_all_projects()
    project_keys = {p["key"] for p in projects}
    project_categories = {
        p["key"]: (p.get("projectCategory") or {}).get("name", "")
        for p in projects
    }
    log.info(f"Found {len(project_keys)} projects: {', '.join(sorted(project_keys))}")

    if len(project_keys) < 2:
        print("Need at least 2 projects to analyze a split.")
        sys.exit(0)

    # Decide whether to include user/permission analysis
    include_users = args.include_users
    if not include_users:
        ans = input("Include user/permission analysis? (y/n) [y]: ").strip().lower()
        include_users = ans != "n"

    # Collect issue-link relationships (kept separate — used for objective scoring)
    raw_edge_weights, issue_counts = collect_relationships(
        jira, project_keys, scan_comments=args.scan_comments
    )

    if not raw_edge_weights:
        print("No cross-project relationships found. Any split works equally well.")
        sys.exit(0)

    # Build augmented edge weights for graph construction
    # (user overlap + category affinity guide initial partitioning strategies)
    aug_edge_weights: dict = dict(raw_edge_weights)

    # Optionally collect user/group data
    user_data: dict = {}
    if include_users:
        log.info("Collecting user and group role assignments …")
        user_data = collect_user_project_roles(jira, project_keys)
        user_overlap_edges = build_user_overlap_edges(user_data)
        if user_overlap_edges and args.user_weight > 0:
            for key, sc in user_overlap_edges.items():
                aug_edge_weights[key] = aug_edge_weights.get(key, 0) + sc * args.user_weight

    cat_edges = build_category_affinity_edges(project_categories, project_keys, args.category_weight)
    for key, w in cat_edges.items():
        aug_edge_weights[key] = aug_edge_weights.get(key, 0) + w

    # Build graph from augmented weights
    G = build_graph(project_keys, aug_edge_weights, issue_counts)
    communities = analyze_communities(G)

    # Find the globally optimal split across all strategies
    log.info("Evaluating partitioning strategies …")
    recommended_sites, best_score, winning_strategy = find_optimal_split(
        G, user_data, project_categories, raw_edge_weights, max_sites=args.max_sites
    )
    recommended_label = f"{len(recommended_sites)}-way"
    cut_weight = best_score["cross_site_links"]

    log.info(
        f"Winner: '{winning_strategy}' — "
        f"disruption={best_score['user_disruption']}  "
        f"cats_split={best_score['categories_split']}  "
        f"cross_links={cut_weight:,}"
    )

    # Report
    print_report(
        G, raw_edge_weights, issue_counts, communities,
        recommended_sites[0],
        recommended_sites[1] if len(recommended_sites) > 1 else set(),
        cut_weight,
    )
    print(f"\n  Strategy selected: {winning_strategy}")
    print(f"  User disruption:   {best_score['user_disruption']} users need 2+ site logins")
    print(f"  Categories split:  {best_score['categories_split']}")
    print(f"  Cross-site links:  {cut_weight:,}")
    if len(recommended_sites) == 3:
        print(f"\n  *** 3-SITE SPLIT — Site C ({len(recommended_sites[2])} projects):")
        for p in sorted(recommended_sites[2]):
            print(f"    - {p}")

    if include_users and user_data:
        impact = analyze_user_disruption_multisite(recommended_sites, user_data)
        print(f"\n--- User Disruption: {impact['disruption_score']} users need access to multiple sites ---")
        if impact["spanning_groups"]:
            print(f"  Groups to duplicate: {', '.join(sorted(impact['spanning_groups']))}")
        print("\n" + "=" * 70)

    # Export raw data
    def _set_to_list(obj):
        """JSON-serialise sets as sorted lists."""
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    output: dict = {
        "edge_weights": {f"{a}|{b}": w for (a, b), w in raw_edge_weights.items()},
        "issue_counts": issue_counts,
        "communities": communities,
        "project_categories": project_categories,
        "recommended_split": {
            "type":             recommended_label,
            "winning_strategy": winning_strategy,
            "sites": {
                chr(65 + i): sorted(s)
                for i, s in enumerate(recommended_sites)
            },
            "score": best_score,
        },
    }
    if include_users and user_data:
        impact = analyze_user_disruption_multisite(recommended_sites, user_data)
        output["user_permission_impact"] = {
            "disruption_score":  impact["disruption_score"],
            "spanning_users":    sorted(impact["spanning_users"]),
            "spanning_groups":   sorted(impact["spanning_groups"]),
            "per_site": {
                lbl: {
                    "users":  sorted(impact["site_users"][i]),
                    "groups": sorted(impact["site_groups"][i]),
                }
                for i, lbl in enumerate(impact["labels"])
            },
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
