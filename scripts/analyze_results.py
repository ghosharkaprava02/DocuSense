"""Plot ablation measurements and paired-bootstrap differences on the demo set."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / "reports"
    frame = pd.read_csv(reports / "ablation.csv")
    melted = frame.melt(id_vars=["engine", "chunk_words"],
                        value_vars=["recall@1", "recall@3", "recall@5", "mrr@5"],
                        var_name="Metric", value_name="Score")
    figure = px.bar(melted, x="chunk_words", y="Score", color="engine", facet_col="Metric",
                    barmode="group", title="DocuSense · Development corpus retrieval ablation",
                    labels={"chunk_words": "Chunk size (words)", "engine": "Retrieval"},
                    template="plotly_dark")
    figure.update_yaxes(range=[0, 1.05])
    figure.write_html(reports / "ablation.html", include_plotlyjs=True)
    intervals = []
    for size in sorted(frame.chunk_words.unique()):
        dense = reports / f"queries_semantic_{size}.csv"
        sparse = reports / f"queries_tfidf_{size}.csv"
        if not dense.exists():
            continue
        pair = pd.read_csv(dense).merge(pd.read_csv(sparse), on="question", suffixes=("_dense", "_tfidf"),
                                       validate="one_to_one")
        delta = (pair["mrr@5_dense"] - pair["mrr@5_tfidf"]).to_numpy()
        rng = np.random.default_rng(42)
        bootstrap = rng.choice(delta, size=(5000, len(delta)), replace=True).mean(axis=1)
        low, high = np.quantile(bootstrap, [.025, .975])
        intervals.append({"chunk_words": int(size), "questions": len(delta),
                          "mrr_difference_dense_minus_tfidf": float(delta.mean()),
                          "bootstrap_95_low": float(low), "bootstrap_95_high": float(high)})
    (reports / "paired_bootstrap.json").write_text(json.dumps({
        "method": "5000 paired question resamples; seed 42; percentile interval",
        "limitation": "Small authored development set; related questions violate independence. "
                      "Intervals describe sensitivity to these questions, not generalization or causality.",
        "results": intervals}, indent=2))
    print(frame.to_string(index=False))
    print("Wrote reports/ablation.html and reports/paired_bootstrap.json")


if __name__ == "__main__":
    main()
