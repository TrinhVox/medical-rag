import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
from generate import run_agent

st.title("ClinicalRAG — PubMed Research Assistant")
st.warning("For research purposes only — not clinical advice.")

# Initialize session state for conversation persistence
if "messages" not in st.session_state:
    st.session_state.messages = []
if "facts" not in st.session_state:
    st.session_state.facts = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask a clinical question..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Searching PubMed and analyzing..."):
            result = run_agent(
                query,
                st.session_state.messages,
                st.session_state.facts
            )
            st.session_state.messages = result["messages"]
            st.session_state.facts = result["facts"]
        st.markdown(result["answer"])
    st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})