"""
Deep Research Agent — Streamlit Frontend
=========================================
Uses FastMCP's own Client to call the MCP server — fixes the 406 error
that occurred when using raw httpx JSON-RPC calls.

Run with:  streamlit run app.py
"""

import time
import asyncio
import streamlit as st
from datetime import datetime

# FastMCP Client — handles all MCP protocol headers automatically
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MCP_BASE = "http://localhost:8000/mcp"

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.5px; }
.tool-badge {
    display: inline-block; font-family: 'Space Mono', monospace;
    font-size: 10px; letter-spacing: 2px; padding: 3px 10px;
    border-radius: 2px; text-transform: uppercase; margin-right: 6px;
}
.badge-search  { background: rgba(59,130,246,0.15);  color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.badge-fetch   { background: rgba(245,158,11,0.15);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-cluster { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.badge-report  { background: rgba(16,185,129,0.15);  color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-error   { background: rgba(255,59,59,0.15);   color: #ff3b3b; border: 1px solid rgba(255,59,59,0.3); }
.step-card {
    background: #0d1117; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px; padding: 16px 20px; margin: 10px 0;
    font-size: 13px; line-height: 1.6;
}
.step-card.active { border-left: 3px solid #f59e0b; }
.step-card.done   { border-left: 3px solid #10b981; }
.step-card.error  { border-left: 3px solid #ff3b3b; }
.mono { font-family: 'Space Mono', monospace; font-size: 11px; color: #6b7280; }
.url-pill {
    display: inline-block; background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 2px;
    padding: 2px 8px; font-size: 11px; margin: 2px;
    font-family: 'Space Mono', monospace; color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Research Agent")
    st.markdown("<div class='mono'>POWERED BY FASTMCP + TAVILY</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### ⚙️ Settings")
    max_results = st.slider("Search results per query", 5, 15, 8)
    n_clusters  = st.slider("Thematic clusters", 2, 6, 4)
    max_urls    = st.slider("URLs to fetch", 3, 8, 5)
    report_fmt  = st.selectbox("Report format", ["markdown", "json"])
    st.divider()
    st.markdown("### 📡 MCP Server")
    server_url = st.text_input("Server URL", value=MCP_BASE)

    if st.button("🔌 Ping Server", use_container_width=True):
        async def _ping(url):
            try:
                transport = StreamableHttpTransport(url)
                async with Client(transport) as c:
                    tools = await c.list_tools()
                    return [t.name for t in tools]
            except Exception as e:
                return str(e)
        result = asyncio.run(_ping(server_url))
        if isinstance(result, list):
            st.success(f"✓ Connected — tools: {', '.join(result)}")
        else:
            st.warning(f"⚠ {result}")

    st.divider()
    st.markdown("### 🔧 Tools")
    for tool, badge in [
        ("search_web","badge-search"),("fetch_and_chunk","badge-fetch"),
        ("cluster_findings","badge-cluster"),("generate_report","badge-report"),
    ]:
        st.markdown(f"<span class='tool-badge {badge}'>{tool}</span>", unsafe_allow_html=True)
    st.divider()
    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_report = None
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:2.2rem;font-weight:800;letter-spacing:-1px;margin-bottom:4px;">
    Deep Research Agent
</h1>
<div class="mono" style="margin-bottom:24px;">
    FASTMCP PYTHON SERVER · TAVILY SEARCH · TF-IDF CLUSTERING
</div>
""", unsafe_allow_html=True)

# ── FastMCP tool caller ───────────────────────────────────────────────────────

async def call_tool_async(tool_name: str, params: dict, url: str) -> dict:
    """Uses FastMCP Client — correctly handles MCP protocol headers (fixes 406)."""
    transport = StreamableHttpTransport(url)
    async with Client(transport) as client:
        result = await client.call_tool(tool_name, params)
    if hasattr(result, "content") and result.content:
        import json
        text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}
    return {"error": "Empty response from tool"}

def call_tool(tool_name: str, params: dict, url: str) -> dict:
    try:
        return asyncio.run(call_tool_async(tool_name, params, url))
    except Exception as e:
        return {"error": str(e)}

def build_queries(topic: str) -> list[str]:
    base = topic.strip().rstrip(".")
    return [base, f"{base} solutions 2025 2026", f"{base} data statistics research"]

# ── Research pipeline ─────────────────────────────────────────────────────────

def run_research_pipeline(topic: str, cfg: dict):
    # Step 1: Search
    all_results = []
    for i, query in enumerate(build_queries(topic), 1):
        ph = st.empty()
        ph.markdown(f"<div class='step-card active'><span class='tool-badge badge-search'>search_web</span>Query {i}/3: <em>{query}</em></div>", unsafe_allow_html=True)
        t0 = time.time()
        res = call_tool("search_web", {"query": query, "max_results": cfg["max_results"]}, cfg["url"])
        elapsed = round(time.time() - t0, 2)
        if "error" in res:
            ph.markdown(f"<div class='step-card error'><span class='tool-badge badge-error'>error</span>{res['error']}</div>", unsafe_allow_html=True)
            yield "error", res, elapsed; return
        hits = res.get("results", [])
        all_results.extend(hits)
        ph.markdown(f"<div class='step-card done'><span class='tool-badge badge-search'>search_web ✓</span><strong>{len(hits)} results</strong> · <em>{query}</em> <span class='mono'>({elapsed}s)</span></div>", unsafe_allow_html=True)
        yield "search", res, elapsed

    # De-duplicate & rank
    seen, ranked = set(), []
    for r in sorted(all_results, key=lambda x: x.get("score", 0), reverse=True):
        if r["url"] not in seen:
            seen.add(r["url"]); ranked.append(r)
    top_urls = [r["url"] for r in ranked[:cfg["max_urls"]]]

    # Step 2: Fetch & Chunk
    ph = st.empty()
    pills = "".join(f"<span class='url-pill'>{u[:55]}…</span>" for u in top_urls)
    ph.markdown(f"<div class='step-card active'><span class='tool-badge badge-fetch'>fetch_and_chunk</span>Fetching {len(top_urls)} URLs…<br>{pills}</div>", unsafe_allow_html=True)
    t0 = time.time()
    fetch_res = call_tool("fetch_and_chunk", {"urls": top_urls, "chunk_size": 400, "max_chunks_per_url": 6}, cfg["url"])
    elapsed = round(time.time() - t0, 2)
    if "error" in fetch_res:
        ph.markdown(f"<div class='step-card error'><span class='tool-badge badge-error'>error</span>{fetch_res['error']}</div>", unsafe_allow_html=True)
        yield "error", fetch_res, elapsed; return
    chunks = fetch_res.get("chunks", [])
    ph.markdown(f"<div class='step-card done'><span class='tool-badge badge-fetch'>fetch_and_chunk ✓</span><strong>{len(chunks)} chunks</strong> from <strong>{fetch_res.get('fetched_urls',0)}</strong> URLs <span class='mono'>({elapsed}s)</span></div>", unsafe_allow_html=True)
    yield "fetch", fetch_res, elapsed
    if not chunks:
        st.error("No content fetched. Check internet connection."); return

    # Step 3: Cluster
    ph = st.empty()
    ph.markdown(f"<div class='step-card active'><span class='tool-badge badge-cluster'>cluster_findings</span>Grouping {len(chunks)} chunks into {cfg['n_clusters']} themes…</div>", unsafe_allow_html=True)
    t0 = time.time()
    cluster_res = call_tool("cluster_findings", {"chunks": chunks, "n_clusters": cfg["n_clusters"]}, cfg["url"])
    elapsed = round(time.time() - t0, 2)
    if "error" in cluster_res:
        ph.markdown(f"<div class='step-card error'><span class='tool-badge badge-error'>error</span>{cluster_res['error']}</div>", unsafe_allow_html=True)
        yield "error", cluster_res, elapsed; return
    clusters = cluster_res.get("clusters", [])
    labels = " · ".join(c.get("label", f"C{i}") for i, c in enumerate(clusters))
    ph.markdown(f"<div class='step-card done'><span class='tool-badge badge-cluster'>cluster_findings ✓</span><strong>{len(clusters)} themes</strong>: {labels} <span class='mono'>({elapsed}s)</span></div>", unsafe_allow_html=True)
    yield "cluster", cluster_res, elapsed

    # Step 4: Generate Report
    ph = st.empty()
    ph.markdown(f"<div class='step-card active'><span class='tool-badge badge-report'>generate_report</span>Synthesising {len(clusters)} clusters…</div>", unsafe_allow_html=True)
    t0 = time.time()
    report_res = call_tool("generate_report", {"topic": topic, "clusters": clusters, "format": cfg["report_format"], "include_sources": True}, cfg["url"])
    elapsed = round(time.time() - t0, 2)
    if "error" in report_res:
        ph.markdown(f"<div class='step-card error'><span class='tool-badge badge-error'>error</span>{report_res['error']}</div>", unsafe_allow_html=True)
        yield "error", report_res, elapsed; return
    ph.markdown(f"<div class='step-card done'><span class='tool-badge badge-report'>generate_report ✓</span><strong>{report_res.get('sections',0)} sections</strong> · <strong>{report_res.get('word_count',0):,} words</strong> <span class='mono'>({elapsed}s)</span></div>", unsafe_allow_html=True)
    yield "report", report_res, elapsed

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "report":
            st.markdown(msg["content"])
            st.download_button("⬇ Download Report", data=msg["content"],
                file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown", key=f"dl_{msg.get('ts', id(msg))}")
        else:
            st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Enter research topic… (e.g. AI energy crisis 2026)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    cfg = {"max_results": max_results, "n_clusters": n_clusters,
           "max_urls": max_urls, "report_format": report_fmt, "url": server_url}

    with st.chat_message("assistant"):
        st.markdown(f"<span class='tool-badge badge-search'>RESEARCH INITIATED</span> Running 4-step pipeline for: **{prompt}**", unsafe_allow_html=True)
        total_time, final_report = 0.0, None
        for step, result, elapsed in run_research_pipeline(prompt, cfg):
            total_time += elapsed
            if step == "report":
                final_report = result.get("report", "")
                st.session_state.last_report = final_report

        if final_report:
            st.divider()
            st.markdown("### 📄 Research Report")
            st.markdown(final_report)
            st.download_button("⬇ Download Report (.md)", data=final_report,
                file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")
            st.markdown(f"<div class='mono'>PIPELINE COMPLETE · {total_time:.1f}s total</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": final_report, "type": "report", "ts": int(time.time())})
        else:
            msg = "⚠️ Pipeline could not complete. Make sure `python server.py` is running and `TAVILY_API_KEY` is set in `.env`."
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})