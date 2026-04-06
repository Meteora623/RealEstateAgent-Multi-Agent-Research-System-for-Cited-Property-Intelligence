from __future__ import annotations

import json

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RealEstate Agent", layout="wide")
st.title("RealEstate Agent")
st.caption("Multi-agent real estate research with citations and trace visibility.")

query = st.text_area(
    "Query",
    value="Which Brooklyn properties appear undervalued based on recent comps?",
    height=120,
)
include_debug = st.toggle("Include retrieval debug", value=True)

if st.button("Run Query"):
    payload = {"query": query, "include_debug": include_debug}
    with st.spinner("Running multi-agent workflow..."):
        response = requests.post(f"{API_BASE}/query/structured", json=payload, timeout=120)

    if response.status_code != 200:
        st.error(f"API error: {response.text}")
    else:
        body = response.json()
        st.subheader("Answer")
        st.write(body["answer"]["concise_answer"])
        st.subheader("Reasoning")
        for item in body["answer"]["reasoning"]:
            st.markdown(f"- {item}")
        st.subheader("Citations")
        for citation in body["answer"]["citations"]:
            st.markdown(
                f"- `{citation['citation_id']}` {citation['doc_id']} / {citation['chunk_id']}: {citation['snippet']}"
            )

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Confidence", body["answer"]["confidence_level"])
        with c2:
            st.metric("Stages", len(body.get("timings_ms", {})))

        st.subheader("Per-Agent Trace")
        st.json(body.get("trace_events", []))
        st.subheader("Timing Breakdown (ms)")
        st.json(body.get("timings_ms", {}))
        if body.get("retrieval_debug"):
            st.subheader("Retrieval Debug")
            st.code(json.dumps(body["retrieval_debug"], indent=2))

