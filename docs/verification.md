# Build verification

Verified on 26 October 2026 using Windows 11 and Python 3.12.14. This report records observed results, not expected values. The core tested package versions are in `requirements-lock.txt`; optional notebook dependencies are in `requirements-dev.txt`.

## Automated checks

`python -m pytest -q`: **44 passed in 23.97 seconds** on the final implementation. Coverage includes PDF extraction and metadata, corrupt/empty/encrypted PDFs, scanned-page errors, stable chunk IDs, overlap and boundaries, cosine normalization and ranking, empty queries, keyword fallback, empty vocabulary, embedding-cache behavior, stale evaluation labels, hand-computed Recall@K/MRR@5, citation rejection, structured-answer validation, Ollama unavailability, clustering validity, and the Streamlit demo/evaluation flow.

The unit suite uses controlled model responses rather than downloading weights or requiring Ollama. Actual model checks below were separate. GitHub Actions runs the same deterministic suite on Linux; its live status is available in the repository's Actions tab.

## App and environment

- Streamlit startup completed and the local health endpoint returned HTTP 200 with `ok`.
- The browser loaded the semantic demo library: 3 documents, 9 pages, 1,411 extracted words, and 13 chunks at size 160 / overlap 30.
- Browser evaluation produced Recall@1/3/5 = 1.000 and MRR@5 = 1.000 on the 18 reviewed demo questions.
- Browser K-Means/PCA completed with 2 clusters (4 and 9 passages) and cosine silhouette approximately 0.153. Clusters are exploratory and not supervised labels.
- Streamlit source watching is disabled in the project config to avoid scanning Transformers' lazy optional modules. Restart the app after code changes.
- VS Code opened the workspace. Python, Pylance, debugger, and Jupyter extensions were already installed. The workspace points at `.venv/Scripts/python.exe` and includes run, test, benchmark, and debug configurations.
- All 4 analysis notebook code cells executed successfully in the project's Python environment. Outputs are saved in the notebook.
- Both four-page PDF guides and the three original demo PDFs were rendered and inspected. No clipped text or overlapping layout was observed in the final render.

## Measured retrieval ablation

The same 18 authored development questions and reviewed page labels were used for every configuration. The table below summarizes `reports/ablation.csv`. Index times exclude model initialization; query times include fresh query encoding, with no query-result or query-embedding cache. Passage embeddings may be reused when identical.

| Engine | Window / overlap | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Mean query ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF | 80 / 15 | 26 | 0.889 | 0.944 | 0.944 | 0.907 | 1.59 |
| MiniLM | 80 / 15 | 26 | 1.000 | 1.000 | 1.000 | 1.000 | 67.70 |
| TF-IDF | 160 / 30 | 13 | 0.889 | 0.944 | 0.944 | 0.907 | 1.66 |
| MiniLM | 160 / 30 | 13 | 1.000 | 1.000 | 1.000 | 1.000 | 49.51 |
| TF-IDF | 220 / 40 | 9 | 0.889 | 0.944 | 0.944 | 0.907 | 3.49 |
| MiniLM | 220 / 40 | 9 | 1.000 | 1.000 | 1.000 | 1.000 | 37.87 |

Dense retrieval ranked the labeled page first for 18/18 questions; TF-IDF did so for 16/18. At window size 160, one TF-IDF miss concerned traceable passage metadata, where the labeled page ranked third; the reproducibility question's labeled page was absent from its top five. Inspect the query-level CSVs to understand vocabulary and ranking behavior.

The paired bootstrap estimated the demo MRR difference at approximately 0.093, with a percentile interval from 0.000 to 0.241 for each configuration. This interval includes zero. Related, authored questions and a tiny corpus limit statistical interpretation. Do not describe the result as a statistically established or generalizable improvement. No held-out evaluation or answer-factuality benchmark was performed.

## Actual local inference

Ollama 0.40.11 was installed from the winget community source, with the installer hash verified by winget. The qwen2.5:1.5b model downloaded successfully. The MiniLM model also downloaded and performed real embedding inference.

`python scripts/smoke_rag.py` completed a real semantic retrieval plus Qwen generation request for: "What is the difference between data drift and concept drift?" The final response had status `generated` with valid source IDs. Measured retrieval was approximately 776 ms and generation approximately 13.0 seconds during concurrent local work. These timings are a single smoke observation, not a controlled performance benchmark.

The generated definition matches the answer-bearing passage in source S1. The model also cited S2 and S3, which were not needed to support this definition. This is a concrete overcitation limitation: source membership validation is working, but precise claim-to-source support still needs human review. The complete answer, timings, and retrieved evidence are preserved in `reports/rag_smoke.json`.

Invalid JSON, unknown references, uncited claims, empty evidence, and unavailable-server behavior are tested separately. Structured claims are rendered with the model-selected citations; the application does not invent source assignments. Empty claim lists produce an explicit abstention. Small-model compliance and factual correctness are not guaranteed.

## Practical limits

This is a local single-user prototype. It has no OCR, authentication, production document isolation, multi-tenant storage, or large-scale vector index. Demo scores are development results. Third-party research papers were not downloaded as part of the final demo verification and are not redistributed. The optional download script and manual PDF upload are available, but their documents need independent relevance labels.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/benchmark.py --semantic
.\.venv\Scripts\python.exe scripts/analyze_results.py
.\.venv\Scripts\python.exe scripts/smoke_rag.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To run the notebook, install `requirements-dev.txt` and select the project environment in VS Code. For cached weights in a deliberately offline session, set `HF_HUB_OFFLINE=1` before starting Python; do this only after the model has been downloaded. Ollama must be running with the named model for generation.
