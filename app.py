#!/usr/bin/env python3
"""
Jira Split Analyzer — Streamlit Web UI

Run locally with:
    streamlit run app.py
"""

import json
import logging
import os
from collections import defaultdict

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from jira_split_analyzer import (
    CHECKPOINT_FILE,
    JiraClient,
    analyze_communities,
    analyze_user_disruption_multisite,
    build_category_affinity_edges,
    build_graph,
    build_user_overlap_edges,
    collect_relationships,
    collect_user_project_roles,
    find_optimal_split,
    score_partition,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Jira Split Analyzer",
    page_icon="🔀",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Logging handler that streams output into a Streamlit container
# ---------------------------------------------------------------------------

class StreamlitLogHandler(logging.Handler):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord):
        self.lines.append(self.format(record))
        self.container.code("\n".join(reversed(self.lines)), language=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env() -> dict:
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    saved: dict = {}
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    saved[k.strip()] = v.strip()
    return saved


def save_env(url: str, email: str, token: str):
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    with open(env_file, "w") as f:
        f.write(f"JIRA_URL={url}\n")
        f.write(f"JIRA_EMAIL={email}\n")
        f.write(f"JIRA_API_TOKEN={token}\n")


SITE_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]   # blue, orange, green
SITE_EMOJIS = ["🔵", "🟠", "🟢"]

def build_plotly_graph(
    G: nx.Graph,
    sites: list[set],
    labels: list[str],
    categories: dict | None = None,
) -> go.Figure:
    pos = nx.spring_layout(G, weight="weight", seed=42)
    max_issues = max((G.nodes[n].get("issues", 0) for n in G.nodes()), default=1) or 1

    # Single edge trace using None separators
    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ex += [x0, x1, None]
        ey += [y0, y1, None]

    edge_trace = go.Scatter(
        x=ex, y=ey,
        mode="lines",
        line=dict(width=1, color="#cccccc"),
        hoverinfo="none",
    )

    node_x_list, node_y_list, node_labels, hover, colors, sizes = [], [], [], [], [], []
    node_site_idx = {}
    for i, site in enumerate(sites):
        for n in site:
            node_site_idx[n] = i

    for node in G.nodes():
        x, y = pos[node]
        node_x_list.append(x)
        node_y_list.append(y)
        node_labels.append(node)
        issues = G.nodes[node].get("issues", 0)
        site_idx = node_site_idx.get(node, 0)
        site_lbl = labels[site_idx] if site_idx < len(labels) else "?"
        cat = (categories or {}).get(node, "")
        cat_line = f"<br>{cat}" if cat else ""
        hover.append(f"<b>{node}</b>{cat_line}<br>Site {site_lbl}<br>{issues:,} issues")
        colors.append(SITE_COLORS[site_idx % len(SITE_COLORS)])
        sizes.append(14 + 26 * (issues / max_issues))

    node_trace = go.Scatter(
        x=node_x_list, y=node_y_list,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker=dict(size=sizes, color=colors, line=dict(width=2, color="white")),
    )

    legend_text = "    ".join(
        f"{SITE_EMOJIS[i % len(SITE_EMOJIS)]} Site {lbl}"
        for i, lbl in enumerate(labels)
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(l=0, r=0, t=40, b=0),
            height=520,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    x=0, y=1.05, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=13),
                    text=legend_text,
                )
            ],
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

saved = load_env()

with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("Connection")
    jira_url = st.text_input(
        "Jira URL", value=saved.get("JIRA_URL", ""),
        placeholder="https://yoursite.atlassian.net",
    )
    email = st.text_input("Email", value=saved.get("JIRA_EMAIL", ""))
    token = st.text_input(
        "API Token", value=saved.get("JIRA_API_TOKEN", ""), type="password",
    )
    save_creds = st.checkbox("Save credentials to .env", value=bool(saved))

    st.subheader("Projects")
    credentials_ready = bool(jira_url and email and token)
    if st.button("🔍 Load Projects", disabled=not credentials_ready, use_container_width=True):
        try:
            with st.spinner("Loading…"):
                _jira = JiraClient(jira_url, email, token)
                _projs = _jira.get_all_projects()
                st.session_state["all_projects"] = sorted(p["key"] for p in _projs)
        except Exception as _exc:
            st.error(f"Could not load projects: {_exc}")

    if "all_projects" in st.session_state:
        _all = st.session_state["all_projects"]
        excluded_projects = st.multiselect(
            "Exclude from analysis",
            options=_all,
            default=st.session_state.get("excluded_projects", []),
            help="Select any projects to exclude from the analysis",
        )
        st.session_state["excluded_projects"] = excluded_projects
        if excluded_projects:
            st.caption(f"Excluding {len(excluded_projects)} project(s): {', '.join(sorted(excluded_projects))}")
    else:
        selected_projects = []
        st.caption("Load projects to choose exclusions, or run directly to include all.")

    st.subheader("Analysis Options")
    scan_comments = st.checkbox("Scan comments (slow)", value=False)
    include_users = st.checkbox("Include user/permission analysis", value=True)
    user_weight = st.slider(
        "User overlap weight", 0.0, 50.0, 10.0, 0.5,
        help="How much shared user/group access influences the split vs issue links",
    )
    category_weight = st.slider(
        "Category affinity weight", 0.0, 500.0, 100.0, 10.0,
        help="How strongly to keep same-category projects on the same site",
    )
    min_site_pct = st.slider(
        "Minimum site size (%)", 5, 45, 15, 5,
        help=(
            "Each site must hold at least this percentage of projects. "
            "Raise it to force a more even split."
        ),
    )
    allow_three_sites = st.checkbox(
        "Allow 3-site split if it reduces user disruption", value=True,
    )

    st.subheader("Checkpoint")
    if os.path.exists(CHECKPOINT_FILE):
        st.info("Checkpoint found — scan will resume from last save.")
        if st.button("🗑️ Clear checkpoint"):
            os.remove(CHECKPOINT_FILE)
            st.success("Checkpoint cleared.")
            st.rerun()
    else:
        st.caption("No checkpoint. A fresh scan will run.")

    st.divider()
    run_button = st.button(
        "▶ Run Analysis", type="primary", use_container_width=True,
        disabled=not (jira_url and email and token),
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🔀 Jira Project Split Analyzer")
st.caption(
    "Analyzes cross-project relationships and recommends the optimal way "
    "to split your Jira instance across two sites — all running locally."
)

if not run_button and "results" not in st.session_state:
    st.info("Enter your Jira credentials in the sidebar and click **▶ Run Analysis** to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

if run_button:
    if save_creds:
        save_env(jira_url, email, token)

    with st.status("Running analysis…", expanded=True) as status:
        log_box = st.empty()
        handler = StreamlitLogHandler(log_box)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        analysis_error: str | None = None

        try:
            jira = JiraClient(jira_url, email, token)

            # Verify auth
            try:
                myself = jira.get("myself")
                st.write(
                    f"✅ Authenticated as **{myself.get('displayName')}** "
                    f"({myself.get('emailAddress')})"
                )
            except Exception as exc:
                analysis_error = f"Authentication failed: {exc}"
                raise

            # Projects
            projects = jira.get_all_projects()
            project_keys = {p["key"] for p in projects}
            project_categories = {
                p["key"]: (p.get("projectCategory") or {}).get("name", "")
                for p in projects
            }

            # Apply exclusions if the user selected any
            _excluded = st.session_state.get("excluded_projects") or []
            if _excluded:
                project_keys = project_keys - set(_excluded)
                st.write(f"Excluding {len(_excluded)} project(s): {', '.join(sorted(_excluded))}")

            if len(project_keys) < 2:
                analysis_error = (
                    f"Found {len(project_keys)} project(s) after exclusions — need at least 2 to analyze a split."
                )
                raise ValueError(analysis_error)
            st.write(f"Analyzing **{len(project_keys)}** projects.")

            # Relationships
            edge_weights, issue_counts = collect_relationships(
                jira, project_keys, scan_comments=scan_comments
            )
            if not edge_weights:
                analysis_error = "No cross-project relationships found. Any split works equally well."
                raise ValueError(analysis_error)

            # User / permission analysis
            user_data: dict = {}
            if include_users:
                user_data = collect_user_project_roles(jira, project_keys)
                user_overlap = build_user_overlap_edges(user_data)

            # Keep raw issue-link weights separate — used for objective scoring
            raw_edge_weights: dict = dict(edge_weights)

            # Augmented weights guide initial partitioning strategies
            aug_weights: dict = dict(raw_edge_weights)
            if include_users and user_overlap and user_weight > 0:
                for key, sc in user_overlap.items():
                    aug_weights[key] = aug_weights.get(key, 0) + sc * user_weight
            cat_edges = build_category_affinity_edges(project_categories, project_keys, category_weight)
            for key, w in cat_edges.items():
                aug_weights[key] = aug_weights.get(key, 0) + w

            # Build graph from augmented weights
            G = build_graph(project_keys, aug_weights, issue_counts)
            communities = analyze_communities(G)

            # Find globally optimal split
            st.write("Evaluating partitioning strategies…")
            max_sites = 3 if allow_three_sites else 2
            recommended_sites, best_score, winning_strategy = find_optimal_split(
                G, user_data, project_categories, raw_edge_weights,
                max_sites=max_sites,
                min_site_fraction=min_site_pct / 100,
            )
            recommended_label = f"{len(recommended_sites)}-site"
            st.write(
                f"**Winner:** {winning_strategy} — "
                f"{best_score['user_disruption']} users spanning sites, "
                f"{best_score['categories_split']} categories split, "
                f"{best_score['cross_site_links']:,} cross-site links."
            )

            # User disruption impact
            user_impact: dict = {}
            if include_users and user_data:
                user_impact = analyze_user_disruption_multisite(recommended_sites, user_data)

            st.session_state["results"] = {
                "G": G,
                "raw_edge_weights": raw_edge_weights,
                "issue_counts": issue_counts,
                "communities": communities,
                "recommended_sites": recommended_sites,
                "recommended_label": recommended_label,
                "best_score": best_score,
                "winning_strategy": winning_strategy,
                "user_data": user_data,
                "user_impact": user_impact,
                "project_keys": project_keys,
                "project_categories": project_categories,
            }
            status.update(label="Analysis complete!", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="Analysis failed.", state="error")
            if analysis_error:
                st.error(analysis_error)
            else:
                st.exception(exc)
            st.stop()
        finally:
            root_logger.removeHandler(handler)

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

results = st.session_state.get("results", {})
if not results:
    st.stop()

G                  = results["G"]
raw_edge_weights   = results["raw_edge_weights"]
issue_counts       = results["issue_counts"]
communities        = results["communities"]
recommended_sites  = results["recommended_sites"]
recommended_label  = results["recommended_label"]
best_score         = results["best_score"]
winning_strategy   = results["winning_strategy"]
user_data          = results["user_data"]
user_impact        = results["user_impact"]
project_keys       = results["project_keys"]
project_categories = results.get("project_categories", {})

site_labels = [chr(65 + i) for i in range(len(recommended_sites))]
cut_weight  = best_score["cross_site_links"]

total_links   = sum(raw_edge_weights.values())
pct_preserved = (1 - cut_weight / total_links) * 100 if total_links else 100.0

# ── Scorecard ────────────────────────────────────────────────────────────────
st.subheader(f"Recommended: {recommended_label} split  ·  Strategy: _{winning_strategy}_")

balance_flag = "✅" if best_score.get("balance_ok", True) else "⚠️"
site_sizes   = [len(s) for s in recommended_sites]

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Projects",              len(project_keys))
c2.metric("Total Issues",          f"{sum(issue_counts.values()):,}")
c3.metric(f"{balance_flag} Smallest Site",
          f"{best_score.get('smallest_site_pct', 0):.0f}%",
          help="Balance — smallest site as % of total projects")
c4.metric("🔴 Users Spanning Sites",
          best_score["user_disruption"],
          help="Priority 1 — users who must log into more than one site")
c5.metric("🟡 Categories Split",
          best_score["categories_split"],
          help="Priority 2 — project categories (departments) broken across sites")
c6.metric("🟢 Cross-site Links",
          f"{cut_weight:,}",
          help="Priority 3 — issue links that would cross site boundaries")
c7.metric("Links Preserved",       f"{pct_preserved:.1f}%")

st.divider()

# Tabs
tab_labels = ["Split Recommendation", "Top Relationships", "Communities"]
if user_impact:
    tab_labels.append("User & Permission Impact")
tabs = st.tabs(tab_labels)

# ── Tab 1: Split recommendation ───────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Recommended {recommended_label} Split  (cut weight: {cut_weight:,})")

    cols = st.columns(len(recommended_sites))
    for i, (site, lbl) in enumerate(zip(recommended_sites, site_labels)):
        with cols[i]:
            issues_total = sum(issue_counts.get(p, 0) for p in site)
            emoji = SITE_EMOJIS[i % len(SITE_EMOJIS)]
            st.markdown(f"**{emoji} Site {lbl}** — {len(site)} projects, {issues_total:,} issues")
            for p in sorted(site):
                cat = project_categories.get(p, "")
                cat_str = f" &nbsp;·&nbsp; _{cat}_" if cat else ""
                st.markdown(f"- `{p}`{cat_str} &nbsp; {issue_counts.get(p, 0):,} issues")

    st.plotly_chart(
        build_plotly_graph(G, recommended_sites, site_labels, project_categories),
        use_container_width=True,
    )

    def site_of(proj):
        for i, s in enumerate(recommended_sites):
            if proj in s:
                return i
        return -1

    broken = [
        {
            "Project A": a, "Category A": project_categories.get(a, ""),
            "Site A": site_labels[site_of(a)] if site_of(a) >= 0 else "?",
            "Project B": b, "Category B": project_categories.get(b, ""),
            "Site B": site_labels[site_of(b)] if site_of(b) >= 0 else "?",
            "Links": w,
        }
        for (a, b), w in sorted(raw_edge_weights.items(), key=lambda x: -x[1])
        if site_of(a) != site_of(b) and site_of(a) >= 0 and site_of(b) >= 0
    ]
    if broken:
        st.subheader(f"Links That Would Cross Sites ({cut_weight:,} total)")
        st.dataframe(pd.DataFrame(broken[:20]), use_container_width=True, hide_index=True)
    else:
        st.success("Clean split — no cross-site links!")

# ── Tab 2: Top relationships ──────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Top 20 Cross-Project Relationships")
    top = sorted(raw_edge_weights.items(), key=lambda x: -x[1])[:20]
    df_top = pd.DataFrame([
        {
            "Project A": a, "Category A": project_categories.get(a, ""),
            "Project B": b, "Category B": project_categories.get(b, ""),
            "Links": w,
        }
        for (a, b), w in top
    ])
    st.dataframe(df_top, use_container_width=True, hide_index=True)

# ── Tab 3: Communities ────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Natural Project Clusters (Louvain)")
    clusters: dict = defaultdict(list)
    for proj, cid in communities.items():
        clusters[cid].append(proj)
    for cid, members in sorted(clusters.items()):
        issues = sum(issue_counts.get(p, 0) for p in members)
        with st.expander(f"Cluster {cid} — {len(members)} projects, {issues:,} issues"):
            for m in sorted(members):
                cat = project_categories.get(m, "")
                cat_str = f" &nbsp;·&nbsp; _{cat}_" if cat else ""
                st.markdown(f"- `{m}`{cat_str} &nbsp; {issue_counts.get(m, 0):,} issues")

# ── Tab 4: User & permission impact ──────────────────────────────────────────
if user_impact:
    with tabs[3]:
        all_site_users  = set().union(*user_impact["site_users"])
        all_site_groups = set().union(*user_impact["site_groups"])
        spanning_users  = user_impact["spanning_users"]
        spanning_groups = user_impact["spanning_groups"]

        # Top metrics
        metric_cols = st.columns(2 + len(recommended_sites))
        metric_cols[0].metric("Total Users",            len(all_site_users))
        metric_cols[1].metric("Users Spanning Sites ⚠️", len(spanning_users))
        for i, lbl in enumerate(site_labels):
            exclusive = user_impact["site_users"][i] - spanning_users
            metric_cols[2 + i].metric(f"Site {lbl} Only", len(exclusive))

        if spanning_users:
            st.warning(
                f"**{len(spanning_users)} user(s) need access to multiple sites** — "
                f"they must be provisioned on each relevant site."
            )

        if spanning_groups:
            st.warning(
                f"**{len(spanning_groups)} group(s) span multiple sites and must be duplicated:**"
            )
            for g in sorted(spanning_groups):
                st.markdown(f"- {g}")

        # Per-site group lists
        for i, lbl in enumerate(site_labels):
            exclusive_groups = user_impact["site_groups"][i] - spanning_groups
            if exclusive_groups:
                with st.expander(f"Groups exclusive to Site {lbl} ({len(exclusive_groups)})"):
                    for g in sorted(exclusive_groups):
                        st.markdown(f"- {g}")

        # Per-project breakdown
        st.subheader("Per-Project Breakdown")
        all_proj = set().union(*recommended_sites)

        def proj_site_label(proj):
            for i, s in enumerate(recommended_sites):
                if proj in s:
                    return site_labels[i]
            return "?"

        rows = [
            {
                "Project":  proj,
                "Category": project_categories.get(proj, ""),
                "Site":     proj_site_label(proj),
                "Users":    len(user_data[proj]["users"]),
                "Groups":   len(user_data[proj]["groups"]),
                "Roles":    ", ".join(sorted(user_data[proj]["roles"].keys())),
            }
            for proj in sorted(all_proj)
            if proj in user_data
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

st.divider()

export: dict = {
    "edge_weights": {f"{a}|{b}": w for (a, b), w in raw_edge_weights.items()},
    "issue_counts": issue_counts,
    "communities": communities,
    "project_categories": project_categories,
    "recommended_split": {
        "type":             recommended_label,
        "winning_strategy": winning_strategy,
        "score":            best_score,
        "sites": {
            lbl: sorted(site)
            for lbl, site in zip(site_labels, recommended_sites)
        },
    },
}
if user_impact:
    export["user_permission_impact"] = {
        "disruption_score":  user_impact["disruption_score"],
        "spanning_users":    sorted(user_impact["spanning_users"]),
        "spanning_groups":   sorted(user_impact["spanning_groups"]),
        "per_site": {
            lbl: {
                "users":  sorted(user_impact["site_users"][i]),
                "groups": sorted(user_impact["site_groups"][i]),
            }
            for i, lbl in enumerate(site_labels)
        },
    }
if user_data:
    export["project_roles"] = {
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

st.download_button(
    "⬇️ Download Results (JSON)",
    data=json.dumps(export, indent=2),
    file_name="jira_split_results.json",
    mime="application/json",
)
