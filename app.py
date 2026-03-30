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
    analyze_user_split_impact,
    build_graph,
    build_user_overlap_edges,
    collect_relationships,
    collect_user_project_roles,
    find_best_bisection,
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
        self.container.code("\n".join(self.lines), language=None)


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


def build_plotly_graph(G: nx.Graph, site_a: set, site_b: set) -> go.Figure:
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

    nx_list, ny_list, labels, hover, colors, sizes = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        nx_list.append(x)
        ny_list.append(y)
        labels.append(node)
        issues = G.nodes[node].get("issues", 0)
        site = "A" if node in site_a else "B"
        hover.append(f"<b>{node}</b><br>Site {site}<br>{issues:,} issues")
        colors.append("#2196F3" if node in site_a else "#FF9800")
        sizes.append(14 + 26 * (issues / max_issues))

    node_trace = go.Scatter(
        x=nx_list, y=ny_list,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker=dict(size=sizes, color=colors, line=dict(width=2, color="white")),
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
                    text="🔵 Site A    🟠 Site B",
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
                if user_overlap and user_weight > 0:
                    merged = dict(edge_weights)
                    for key, score in user_overlap.items():
                        merged[key] = merged.get(key, 0) + score * user_weight
                    edge_weights = merged

            # Graph + bisection
            G = build_graph(project_keys, edge_weights, issue_counts)
            communities = analyze_communities(G)
            site_a, site_b, cut_weight = find_best_bisection(G)

            # User impact
            user_impact: dict = {}
            if include_users and user_data:
                user_impact = analyze_user_split_impact(site_a, site_b, user_data)

            st.session_state["results"] = {
                "G": G,
                "edge_weights": edge_weights,
                "issue_counts": issue_counts,
                "communities": communities,
                "site_a": site_a,
                "site_b": site_b,
                "cut_weight": cut_weight,
                "user_data": user_data,
                "user_impact": user_impact,
                "project_keys": project_keys,
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

G              = results["G"]
edge_weights   = results["edge_weights"]
issue_counts   = results["issue_counts"]
communities    = results["communities"]
site_a         = results["site_a"]
site_b         = results["site_b"]
cut_weight     = results["cut_weight"]
user_data      = results["user_data"]
user_impact    = results["user_impact"]
project_keys   = results["project_keys"]

total_links    = sum(edge_weights.values())
pct_preserved  = (1 - cut_weight / total_links) * 100 if total_links else 100.0

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Projects",            len(project_keys))
c2.metric("Total Issues",        f"{sum(issue_counts.values()):,}")
c3.metric("Cross-project Links", f"{total_links:,}")
c4.metric("Links Preserved",     f"{pct_preserved:.1f}%")

st.divider()

# Tabs
tab_labels = ["Split Recommendation", "Top Relationships", "Communities"]
if user_impact:
    tab_labels.append("User & Permission Impact")
tabs = st.tabs(tab_labels)

# ── Tab 1: Split recommendation ───────────────────────────────────────────────
with tabs[0]:
    st.subheader(f"Recommended 2-Way Split  (cut weight: {cut_weight:,})")

    col_a, col_b = st.columns(2)
    with col_a:
        issues_a = sum(issue_counts.get(p, 0) for p in site_a)
        st.markdown(f"**🔵 Site A** — {len(site_a)} projects, {issues_a:,} issues")
        for p in sorted(site_a):
            st.markdown(f"- `{p}` &nbsp; {issue_counts.get(p, 0):,} issues")
    with col_b:
        issues_b = sum(issue_counts.get(p, 0) for p in site_b)
        st.markdown(f"**🟠 Site B** — {len(site_b)} projects, {issues_b:,} issues")
        for p in sorted(site_b):
            st.markdown(f"- `{p}` &nbsp; {issue_counts.get(p, 0):,} issues")

    st.plotly_chart(build_plotly_graph(G, site_a, site_b), use_container_width=True)

    broken = [
        {"Project A": a, "Project B": b, "Links": w}
        for (a, b), w in sorted(edge_weights.items(), key=lambda x: -x[1])
        if (a in site_a and b in site_b) or (a in site_b and b in site_a)
    ]
    if broken:
        st.subheader(f"Links That Would Cross Sites ({cut_weight:,} total)")
        st.dataframe(pd.DataFrame(broken[:20]), use_container_width=True, hide_index=True)
    else:
        st.success("Clean split — no cross-site links!")

# ── Tab 2: Top relationships ──────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Top 20 Cross-Project Relationships")
    top = sorted(edge_weights.items(), key=lambda x: -x[1])[:20]
    df_top = pd.DataFrame(
        [{"Project A": a, "Project B": b, "Links": w} for (a, b), w in top]
    )
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
                st.markdown(f"- `{m}` &nbsp; {issue_counts.get(m, 0):,} issues")

# ── Tab 4: User & permission impact ──────────────────────────────────────────
if user_impact:
    with tabs[3]:
        total_users = (
            len(user_impact["users_site_a_only"])
            + len(user_impact["users_site_b_only"])
            + len(user_impact["users_on_both"])
        )
        total_groups = (
            len(user_impact["groups_site_a_only"])
            + len(user_impact["groups_site_b_only"])
            + len(user_impact["groups_on_both"])
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Users",          total_users)
        m2.metric("Site A Only",          len(user_impact["users_site_a_only"]))
        m3.metric("Site B Only",          len(user_impact["users_site_b_only"]))
        m4.metric("Spanning Both ⚠️",     len(user_impact["users_on_both"]))
        m5.metric("Total Groups",         total_groups)
        m6.metric("Groups to Duplicate ⚠️", len(user_impact["groups_on_both"]))

        if user_impact["groups_on_both"]:
            st.warning(
                f"**{len(user_impact['groups_on_both'])} group(s) span both sites "
                f"and must be duplicated:**"
            )
            for g in sorted(user_impact["groups_on_both"]):
                st.markdown(f"- {g}")

        if user_impact["groups_site_a_only"]:
            with st.expander(f"Groups staying on Site A ({len(user_impact['groups_site_a_only'])})"):
                for g in sorted(user_impact["groups_site_a_only"]):
                    st.markdown(f"- {g}")

        if user_impact["groups_site_b_only"]:
            with st.expander(f"Groups staying on Site B ({len(user_impact['groups_site_b_only'])})"):
                for g in sorted(user_impact["groups_site_b_only"]):
                    st.markdown(f"- {g}")

        st.subheader("Per-Project Breakdown")
        rows = [
            {
                "Project": proj,
                "Site": "A" if proj in site_a else "B",
                "Users": len(user_data[proj]["users"]),
                "Groups": len(user_data[proj]["groups"]),
                "Roles": ", ".join(sorted(user_data[proj]["roles"].keys())),
            }
            for proj in sorted(site_a | site_b)
            if proj in user_data
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

st.divider()

export: dict = {
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
    export["user_permission_impact"] = {k: sorted(v) for k, v in user_impact.items()}
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
