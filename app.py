"""DocuSense research workstation. Run: python -m streamlit run app.py."""
from pathlib import Path
import json
import os
from time import perf_counter

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import cluster_chunks, document_stats, keywords
from src.chunking import chunk_document
from src.embeddings import Embedder, MODEL_NAME
from src.evaluation import Example, evaluate
from src.pdf_parser import parse_pdf
from src.rag import generate_answer
from src.retrieval import SearchIndex

ROOT = Path(__file__).parent
st.set_page_config(page_title="DocuSense | Document intelligence", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container {padding-top:2.4rem; max-width:1440px}
div[data-testid="stMetric"] {background:#141f32;border:1px solid #26354b;border-radius:12px;padding:16px}
.eyebrow {color:#4dd8c0;font-size:.78rem;letter-spacing:.17em;font-weight:700}
.hero {font-size:3rem;font-weight:750;letter-spacing:-.04em;margin:0}
.sub {color:#a4b4ca;max-width:800px;margin:.3rem 0 1.8rem}
.stTabs [data-baseweb="tab-list"] {gap:24px;border-bottom:1px solid #26354b}
.stTabs [data-baseweb="tab"] {height:52px}
div[data-testid="stExpander"] {border-color:#26354b}
</style>""", unsafe_allow_html=True)

for key, value in {"documents": [], "chunks": [], "index": None, "embedder": None,
                   "history": [], "annotations": [], "evaluation": None,
                   "index_ms": None, "cluster_result": None, "notices": []}.items():
    if key not in st.session_state:
        st.session_state[key] = value
state = st.session_state


def reset_results():
    state.history = []
    state.annotations = []
    state.evaluation = None
    state.cluster_result = None


def build_index(files, mode, size, overlap):
    documents, chunks, errors, seen = [], [], [], set()
    for name, content in files:
        try:
            doc = parse_pdf(content, name)
            if doc.id in seen:
                errors.append(f"{name}: duplicate content skipped.")
                continue
            seen.add(doc.id)
            documents.append(doc)
            chunks.extend(chunk_document(doc, size, overlap))
            errors.extend(f"{name}: {warning}" for warning in doc.warnings)
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
    if not chunks:
        for error in errors:
            st.warning(error)
        st.error("No text could be indexed. Upload text-based PDFs.")
        return False
    if len(chunks) > 10000:
        st.error("This local prototype supports at most 10,000 chunks per library.")
        return False
    start = perf_counter()
    embedder = None
    if mode == "Semantic · MiniLM":
        try:
            if state.embedder is None:
                state.embedder = Embedder()
            embedder = state.embedder
            index = SearchIndex(chunks, embedder)
        except Exception as exc:
            errors.append(f"Semantic indexing unavailable ({type(exc).__name__}). "
                          "Using explicitly labeled keyword TF-IDF search. "
                          "Check network/model availability and rebuild for semantic search.")
            try:
                index = SearchIndex(chunks)
            except ValueError as error:
                st.error(str(error))
                return False
    else:
        try:
            index = SearchIndex(chunks)
        except ValueError as error:
            st.error(str(error))
            return False
    state.documents, state.chunks, state.index = documents, chunks, index
    state.index_ms = (perf_counter() - start) * 1000
    state.notices = errors
    reset_results()
    return True


def show_evidence(hits):
    for i, hit in enumerate(hits, 1):
        with st.expander(f"[S{i}] {hit.chunk.filename} · p. {hit.chunk.page} · score {hit.score:.3f}"):
            st.text(hit.chunk.text)
            st.caption(f"Chunk: {hit.chunk.id} | Page ID: {hit.chunk.document_id}:p{hit.chunk.page}")


with st.sidebar:
    st.markdown("### ◈ DocuSense")
    st.caption("LOCAL DOCUMENT INTELLIGENCE")
    st.divider()
    st.markdown("**Workspace status**")
    st.caption("Index: " + (state.index.mode if state.index else "not built"))
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Generation: Ollama / qwen2.5:1.5b")
    st.info("Uploads and embeddings stay in this session. Model weights are cached locally. "
            "First semantic indexing downloads the embedding model.")
    if st.button("Clear workspace", use_container_width=True):
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()
    st.caption("Local prototype · No API keys · No telemetry configured")

st.markdown('<div class="eyebrow">RESEARCH / RETRIEVE / UNDERSTAND</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Your documents. Connected.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Explore evidence, uncover patterns, and measure retrieval quality. '
            'A local research workstation for document intelligence.</div>', unsafe_allow_html=True)

stats = document_stats(state.documents, state.chunks)
metrics = st.columns(4)
for col, label, value in zip(metrics, ["Documents", "Pages", "Extracted words", "Indexed passages"],
                             [len(state.documents), int(stats.Pages.sum()),
                              int(stats.Words.sum()), len(state.chunks)]):
    col.metric(label, f"{value:,}")

library, ask, analytics, evaluation_tab = st.tabs([
    "01  Document Library", "02  Ask AI", "03  Document Analytics", "04  Retrieval Evaluation"])

with library:
    st.subheader("Build your evidence library")
    st.caption("Text PDFs only · 25 MB / file · 500 pages / file · 10,000 chunks / library")
    uploads = st.file_uploader("Drop research papers, reports, or notes", type="pdf", accept_multiple_files=True)
    left, mid, right = st.columns([2, 1, 1])
    mode = left.selectbox("Retrieval engine", ["Semantic · MiniLM", "Keyword · TF-IDF baseline"])
    size = mid.slider("Chunk size (words)", 40, 220, 160, 10)
    overlap = right.slider("Overlap (words)", 0, min(100, size - 1), min(30, size - 1), 1)
    st.caption("Word windows are page-bounded. MiniLM truncates inputs beyond its token budget; "
               "technical words may split into several tokens. Compare chunk sizes in evaluation.")
    col1, col2 = st.columns(2)
    if col1.button("Index uploaded PDFs", type="primary", disabled=not uploads, use_container_width=True):
        with st.spinner("Extracting text and building index…"):
            built = build_index([(f.name, f.getvalue()) for f in uploads], mode, size, overlap)
        if built:
            st.rerun()
    demo_files = sorted((ROOT / "examples" / "demo").glob("*.pdf"))
    if col2.button("Load reproducible demo", disabled=not demo_files, use_container_width=True):
        with st.spinner("Indexing the authored demo corpus…"):
            built = build_index([(p.name, p.read_bytes()) for p in demo_files], mode, size, overlap)
        if built:
            st.rerun()
    for notice in state.notices:
        st.warning(notice)
    if state.documents:
        st.dataframe(stats, hide_index=True, use_container_width=True)
        st.caption(f"Last indexing: {state.index_ms / 1000:.2f}s · Active engine: {state.index.mode}")
        manifest = pd.DataFrame([{"filename": c.filename, "page": c.page,
                                 "page_id": f"{c.document_id}:p{c.page}",
                                 "chunk_id": c.id, "text": c.text} for c in state.chunks])
        st.download_button("Export source manifest (contains extracted text)",
                           manifest.to_csv(index=False).encode(), "docusense_manifest.csv", "text/csv")
    else:
        st.info("Start with the included original demo PDFs, or upload your own text PDFs. "
                "Research paper download commands are in the README.")

with ask:
    st.subheader("Ask with evidence")
    if state.index is None:
        st.info("Build a library in the first tab to search and ask questions.")
    else:
        st.caption(f"Active retrieval: {state.index.mode}. Scores are similarities, not probabilities.")
        c1, c2 = st.columns([1, 3])
        k = c1.slider("Top-k passages", 1, 8, 5)
        answer_mode = c2.radio("Response", ["Search evidence", "Generate local RAG answer"], horizontal=True)
        with st.form("question"):
            question = st.text_input("Question", placeholder="How do we detect data drift after deployment?")
            submitted = st.form_submit_button("Explore evidence", type="primary")
        if submitted and question.strip():
            with st.spinner("Retrieving evidence…"):
                hits, latency = state.index.search(question, k)
            answer = None
            if answer_mode == "Generate local RAG answer":
                with st.spinner("Waiting for local Ollama inference…"):
                    answer = generate_answer(question, hits)
            state.history.append({"question": question, "hits": hits,
                                  "retrieval_ms": latency, "answer": answer})
            state.history = state.history[-10:]
        for item in reversed(state.history):
            with st.container(border=True):
                st.markdown(f"**{item['question']}**")
                answer = item["answer"]
                if answer:
                    if answer.text:
                        st.markdown(answer.text)
                    st.caption(f"Generation: {answer.generation_ms:.0f} ms · {answer.status}")
                    st.info(answer.message)
                st.caption(f"Retrieval: {item['retrieval_ms']:.1f} ms · {len(item['hits'])} passages")
                if not item["hits"]:
                    st.info("No matching evidence found.")
                show_evidence(item["hits"])
        st.caption("Retrieved passages are untrusted data. Source validation checks IDs only; "
                   "it cannot establish factual correctness or eliminate prompt injection.")

with analytics:
    st.subheader("See the shape of your corpus")
    if not state.chunks:
        st.info("Upload and index documents to calculate real corpus statistics.")
    else:
        st.metric("Average passage length (words)", f"{np.mean([len(c.text.split()) for c in state.chunks]):.1f}")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(stats, x="Document", y="Words", color="Document",
                              title="Extracted words by document"), use_container_width=True)
        lengths = pd.DataFrame({"Words per page": [len(p.text.split()) for d in state.documents for p in d.pages]})
        c2.plotly_chart(px.histogram(lengths, x="Words per page", nbins=20,
                                    title="Page length distribution"), use_container_width=True)
        freq = pd.DataFrame(keywords(" ".join(p.text for d in state.documents for p in d.pages)),
                            columns=["Term", "Frequency"])
        st.plotly_chart(px.bar(freq, x="Frequency", y="Term", orientation="h",
                               title="Frequent terms · English stopwords removed").update_layout(
                                   yaxis={"categoryorder": "total ascending"}), use_container_width=True)
        st.subheader("Exploratory topics")
        if state.index.vectors is None:
            st.info("Embedding clustering requires the semantic index. Keyword search does not create embeddings.")
        elif len(state.chunks) < 3:
            st.info("At least 3 passages are needed for clustering and PCA.")
        else:
            upper = min(10, len(state.chunks) - 1, len(np.unique(state.index.vectors, axis=0)))
            if upper < 2:
                st.info("Not enough distinct embeddings for multiple clusters.")
            else:
                clusters = st.selectbox("Number of clusters", list(range(2, upper + 1)))
                if st.button("Run K-Means + PCA"):
                    with st.spinner("Clustering embeddings…"):
                        state.cluster_result = cluster_chunks(state.chunks, state.index.vectors, clusters)
                result = state.cluster_result
                if result:
                    points = pd.DataFrame(result["coordinates"], columns=["PC1", "PC2"])
                    points["Topic"] = [str(x) for x in result["labels"]]
                    points["Document"] = [c.filename for c in state.chunks]
                    points["Page"] = [c.page for c in state.chunks]
                    st.plotly_chart(px.scatter(points, x="PC1", y="PC2", color="Topic",
                                               hover_data=["Document", "Page"], title="PCA projection"),
                                    use_container_width=True)
                    if result["silhouette"] is not None:
                        st.metric("Cosine silhouette · stratified sample, <=200 / cluster",
                                  f"{result['silhouette']:.3f}")
                    for topic in result["topics"]:
                        representative = topic["representative"]
                        with st.expander(f"Topic {topic['cluster']} · {topic['size']} passages · " +
                                         ", ".join(w for w, _ in topic["keywords"][:5])):
                            st.write(representative.text)
                            st.caption(f"{representative.filename} · p. {representative.page} · nearest centroid passage")
                    st.caption("Clusters are exploratory, not labeled classes. PCA distorts distances; "
                               "silhouette measures separation, not semantic truth.")

with evaluation_tab:
    st.subheader("Measure retrieval before making claims")
    st.caption("Manual relevance labels · Macro Recall@K · MRR@5 · Actual timed searches")
    if not state.chunks:
        st.info("Build an index, then annotate relevant pages or import verified labels.")
    else:
        valid_ids = {"page": {f"{c.document_id}:p{c.page}" for c in state.chunks},
                     "chunk": {c.id for c in state.chunks}}
        with st.expander("Annotate a question against the current corpus", expanded=not state.annotations):
            label_question = st.text_input("Evaluation question")
            options = {}
            for document in state.documents:
                for page in document.pages:
                    pid = f"{document.id}:p{page.number}"
                    if pid in valid_ids["page"]:
                        options[pid] = f"{document.filename} · page {page.number}"
            selected = st.multiselect("Relevant pages (inspect the full page before labeling)",
                                      list(options), format_func=options.get)
            if selected:
                for document in state.documents:
                    for page in document.pages:
                        if f"{document.id}:p{page.number}" in selected:
                            st.text(page.text)
            if st.button("Add verified label", disabled=not (label_question.strip() and selected)):
                state.annotations.append(Example(label_question.strip(), frozenset(selected)))
                state.evaluation = None
        labels_file = st.file_uploader("Import labels JSON", type="json")
        if st.button("Load imported labels", disabled=labels_file is None):
            try:
                raw = json.loads(labels_file.getvalue())
                examples = [Example(x["question"], frozenset(x["relevant_ids"]), x.get("unit", "page")) for x in raw]
                if len(examples) > 200:
                    raise ValueError("Maximum 200 examples per run.")
                state.annotations = examples
                state.evaluation = None
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"Invalid label file: {exc}")
        demo_labels = ROOT / "examples" / "demo" / "labels.json"
        if st.button("Load reviewed demo labels", disabled=not demo_labels.exists()):
            raw = json.loads(demo_labels.read_text())
            state.annotations = [Example(x["question"], frozenset(x["relevant_ids"]), x["unit"]) for x in raw]
            state.evaluation = None
        if state.annotations:
            st.dataframe([{"question": x.question, "unit": x.unit,
                           "relevant_ids": "|".join(sorted(x.relevant_ids))} for x in state.annotations],
                         hide_index=True, use_container_width=True)
            export_labels = [{"question": x.question, "unit": x.unit,
                              "relevant_ids": sorted(x.relevant_ids)} for x in state.annotations]
            st.download_button("Export annotations", json.dumps(export_labels, indent=2),
                               "docusense_labels.json", "application/json")
            if st.button("Run retrieval evaluation", type="primary"):
                try:
                    with st.spinner("Running labeled queries…"):
                        state.evaluation = evaluate(state.annotations, state.index.search, valid_ids)
                except ValueError as exc:
                    st.error(str(exc))
        else:
            st.info("No ground truth labels yet. Add a question and relevant pages above. "
                    "Demo labels apply only to the included authored demo corpus.")
        if state.evaluation:
            frame, summary = state.evaluation
            cols = st.columns(4)
            for col, label in zip(cols, ["recall@1", "recall@3", "recall@5", "mrr@5"]):
                col.metric(label, f"{summary[label]:.3f}")
            st.caption(f"Engine: {state.index.mode} · {len(frame)} questions · "
                       f"Mean retrieval latency: {summary['retrieval_ms']:.1f} ms")
            st.dataframe(frame, hide_index=True, use_container_width=True)
            st.download_button("Export query results CSV", frame.to_csv(index=False).encode(),
                               "docusense_evaluation.csv", "text/csv")
        generation = [item['answer'].generation_ms for item in state.history if item['answer']]
        if generation:
            st.metric("Mean generation attempt latency (session)", f"{np.mean(generation):.0f} ms")
            st.caption("Includes unsuccessful attempts. See each Ask AI result for generation status.")
        else:
            st.caption("Generation latency: not measured in this session. Use local RAG in Ask AI.")
        st.warning("Retrieval relevance is not answer factuality. Demo questions are a small development set, "
                   "not a held-out benchmark. Do not generalize scores to unseen documents.")
