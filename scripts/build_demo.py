"""Generate original, deterministic demo PDFs and source-grounded page labels.

No research-paper text is redistributed. Each question is paired explicitly with
the authored page containing its answer; see annotation_review.md.
"""
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, PageBreak

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pdf_parser import parse_pdf

CORPUS = {
    "ml_systems.pdf": [
        ("Reliable training data", [
            "The training set teaches model parameters, the validation set guides model selection, and the test set estimates final performance. Split the dataset before learning any preprocessing statistics. Fit imputation, scaling, and feature selection on the training partition only. Reuse those fitted transformations for validation and test examples.",
            "Leakage occurs when information unavailable at prediction time enters model development. A patient appearing in both training and test records can leak identity. Use group-aware splitting for repeated entities. For forecasting, train on earlier observations and evaluate on later observations. A random split may produce an optimistic estimate when temporal dependence is present.",
            "Record dataset provenance, label definitions, missing-value rates, class frequencies, and the split seed. Inspect anomalous records before removing them. Compare a simple baseline before developing a complex model. A held-out test set should remain untouched until the model selection process is complete."]),
        ("Classification evaluation", [
            "Accuracy can conceal poor minority-class performance. Precision is the proportion of predicted positives that are actually positive. Recall is the proportion of actual positives recovered by the model. The F1 score is the harmonic mean of precision and recall. Choose a decision threshold using the validation set and the operational cost of false positives and false negatives.",
            "Use a confusion matrix to inspect error types. For imbalanced classification, precision-recall curves can be more informative than accuracy. Compare metrics by relevant data slices and report sample counts. Small slices have high uncertainty. Bootstrap confidence intervals can describe sampling uncertainty, although correlated observations require group-aware resampling.",
            "A probability estimate is calibrated when events assigned a probability near 0.8 occur about eighty percent of the time. Inspect calibration curves when predicted probabilities drive decisions. Discrimination and calibration answer different questions. Report both where appropriate, and document the dataset used for each measurement."]),
        ("Monitoring after deployment", [
            "Monitor input feature distributions, missing-value rates, inference latency, and model errors after deployment. Data drift describes changes in the distribution of inputs. Concept drift describes changes in the relationship between inputs and the target. A change in inputs alone does not prove that predictive quality has degraded.",
            "Compare reference and recent windows using distribution summaries, population stability measures, or statistical tests. These are alerts for investigation, not automatic proof that retraining is needed. Once delayed labels become available, calculate performance on recent observations and compare with the training-era evaluation.",
            "Log the model version and preprocessing version with each inference. Establish an alert owner and a rollback path. Retraining should pass the same validation checks as the original model. Monitoring without a response process provides little operational value."])
    ],
    "retrieval_lab.pdf": [
        ("Dense and lexical retrieval", [
            "Lexical retrieval compares words appearing in a question and a passage. TF-IDF increases the weight of terms frequent in a passage but uncommon across the corpus. A lexical baseline is inexpensive and interpretable, but it can miss passages that express the same idea with different vocabulary.",
            "Dense retrieval maps text into a vector representation using a pretrained encoder. Normalize both question and passage vectors to unit length. Their dot product then equals cosine similarity. A higher cosine score indicates greater alignment in the representation space; it is not a calibrated probability that a passage answers the question.",
            "Evaluate the lexical baseline and the dense system on the same relevance labels. Specialized terminology, identifiers, and exact names may favor lexical retrieval, while paraphrased questions can benefit from semantic representations. A dense model pretrained on general text may underperform on unfamiliar domains."]),
        ("Chunking and evidence provenance", [
            "Chunking divides extracted text into windows that fit a retrieval encoder and generation context. Overlap repeats boundary words between adjacent windows, helping preserve local continuity. Larger windows include more context but can dilute a short relevant statement. Smaller windows improve granularity but can separate an answer from its explanation.",
            "Keep windows within a page so every passage retains an unambiguous page reference. A source record includes document content hash, page number, word offsets, and chunk ID. Stable chunk IDs should depend on content and chunking parameters. Changing the chunk size can change chunk IDs even when the original PDF is unchanged.",
            "Word counts are not tokenizer counts. Technical words may split into multiple subwords, and an embedding model may truncate a long window. Measure retrieval across several window sizes rather than assuming that one setting is universally optimal. Page labels remain usable across chunking experiments when the PDF bytes remain unchanged."]),
        ("Grounded generation and abstention", [
            "Retrieval-augmented generation first retrieves passages and then asks a language model to answer from those passages. Give each retrieved passage a source label. Ask the model to cite those labels for factual claims and to state when evidence is insufficient. Keep source passages visible so readers can inspect support.",
            "Document text is untrusted data. A passage can contain instructions that attempt to redirect the model. Place behavioral instructions outside the evidence and state that evidence instructions must not be followed. This separation reduces some risks but does not eliminate prompt injection.",
            "Citation validation checks that referenced IDs were supplied to the model. It does not establish that a cited passage entails a claim. An answer can cite a valid source and still be wrong. Withhold answers with unknown citation labels, and evaluate claim support separately with human review. If local inference is unavailable, show retrieved evidence without inventing an answer."])
    ],
    "evaluation_notes.pdf": [
        ("Retrieval metrics", [
            "A retrieval evaluation set contains questions and manually verified relevant source IDs. Recall at K is the number of distinct relevant IDs in the first K results divided by the number of annotated relevant IDs. Several chunks from the same relevant page count once in the numerator when the relevance unit is a page, but they still occupy retrieval ranks.",
            "Reciprocal rank is one divided by the rank of the first relevant result. If no relevant result appears within the evaluated depth, reciprocal rank is zero. Mean reciprocal rank averages this value across questions. DocuSense reports MRR at depth five, called MRR@5, rather than an unrestricted full-ranking MRR.",
            "Macro averaging gives each question equal weight. Incomplete relevance annotations can underestimate recall, while easy development questions can overestimate generalization. Always report the relevance unit, cutoff depth, question count, corpus identity, and whether the questions were used for tuning."]),
        ("Experiments and reproducibility", [
            "An ablation changes a controlled design choice to understand its contribution. Compare TF-IDF and dense retrieval while keeping questions and page labels fixed. Next vary chunk size and overlap, and record indexing time, query latency, Recall@K, and MRR@5. Do not describe development-set improvements as held-out test gains.",
            "Track the model name, dependency versions, random seed, and PDF content hashes. Separate cold model-loading time from warm query latency. A warm embedding cache can make later index builds faster than the first one. The order of experiments therefore matters for timing comparisons.",
            "Reserve independent questions and additional documents for a final evaluation. Avoid choosing all settings by repeatedly consulting a small final test set. State hardware and execution conditions before comparing latency measurements from different machines. Reproducible code does not guarantee identical wall-clock timings."]),
        ("Exploratory topics", [
            "K-Means groups vectors by minimizing squared Euclidean distance to cluster centers. Fix a random seed and use multiple initializations for reproducibility. A representative passage can be selected as the chunk closest to its cluster center. Descriptive keywords summarize frequent terms within the cluster; they are not supervised class labels.",
            "Principal component analysis projects high-dimensional embeddings onto directions of largest variance. A two-dimensional plot helps inspect broad patterns, but projection can distort distances. Nearby points in the plot are not proof of identical topics. The original embedding space remains the basis for clustering.",
            "Silhouette compares within-cluster cohesion with separation from other clusters. It requires at least two clusters and fewer clusters than observations. Compute the score only when valid, and use a bounded sample for larger datasets. High separation does not prove that clusters correspond to useful human categories."])
    ]
}

QUESTIONS = [
    ("ml_systems.pdf", 1, "When should preprocessing statistics be fitted to prevent leakage?"),
    ("ml_systems.pdf", 1, "How should repeated patients be split between training and test sets?"),
    ("ml_systems.pdf", 2, "What is the difference between precision and recall?"),
    ("ml_systems.pdf", 2, "What does probability calibration mean?"),
    ("ml_systems.pdf", 3, "How do data drift and concept drift differ?"),
    ("ml_systems.pdf", 3, "What should be monitored after model deployment?"),
    ("retrieval_lab.pdf", 1, "Why does a normalized vector dot product equal cosine similarity?"),
    ("retrieval_lab.pdf", 1, "What are the limitations of a lexical TF-IDF baseline?"),
    ("retrieval_lab.pdf", 2, "Why does overlap help with chunk boundaries?"),
    ("retrieval_lab.pdf", 2, "Which metadata makes a passage traceable to its source?"),
    ("retrieval_lab.pdf", 3, "Does validating source IDs prove answer factuality?"),
    ("retrieval_lab.pdf", 3, "How should the application behave when local inference is unavailable?"),
    ("evaluation_notes.pdf", 1, "How is page-level Recall@K calculated with duplicate page results?"),
    ("evaluation_notes.pdf", 1, "How is MRR@5 computed when no relevant result is found?"),
    ("evaluation_notes.pdf", 2, "What information is needed to reproduce a retrieval experiment?"),
    ("evaluation_notes.pdf", 2, "Why can warm caches affect index timing comparisons?"),
    ("evaluation_notes.pdf", 3, "What does a nearest-centroid representative passage mean?"),
    ("evaluation_notes.pdf", 3, "When is silhouette score mathematically valid?")
]


def main():
    output = ROOT / "examples" / "demo"
    output.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 11
    styles["BodyText"].leading = 17
    styles["BodyText"].spaceAfter = 18
    documents = {}
    for filename, pages in CORPUS.items():
        story = []
        for i, (heading, paragraphs) in enumerate(pages):
            if i:
                story.append(PageBreak())
            story.extend([Paragraph("DOCUSENSE / ORIGINAL DEMO CORPUS", styles["Heading3"]),
                          Spacer(1, 12), Paragraph(heading, styles["Title"]), Spacer(1, 12)])
            story.extend(Paragraph(escape(p), styles["BodyText"]) for p in paragraphs)
        def footer(canvas, doc):
            canvas.setFillColor(colors.HexColor("#52647a"))
            canvas.setFont("Helvetica", 9)
            canvas.drawString(54, 36, "Original educational content. Development evaluation only.")
            canvas.drawRightString(558, 36, f"{doc.page}")
        pdf = output / filename
        SimpleDocTemplate(str(pdf), pagesize=(612, 792), leftMargin=54, rightMargin=54,
                          topMargin=50, bottomMargin=58, invariant=1,
                          title=filename.replace("_", " ").replace(".pdf", "")).build(
                              story, onFirstPage=footer, onLaterPages=footer)
        documents[filename] = parse_pdf(pdf.read_bytes(), filename)
        assert len(documents[filename].pages) == 3, "Demo headings must map one-to-one to pages."
    labels = [{"question": q, "unit": "page", "relevant_ids": [f"{documents[f].id}:p{p}"],
               "source_file": f, "source_page": p} for f, p, q in QUESTIONS]
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    review = ["# Demo annotation review", "", "Original educational corpus; not a public benchmark.",
              "Each question was authored against and reviewed with the named source page. "
              "These are narrow answer-bearing page labels, not exhaustive relevance judgments. "
              "Questions are development examples; no held-out generalization claim is justified.", ""]
    for file, page, question in QUESTIONS:
        review.extend([f"## {question}", f"Source: {file}, page {page}.",
                       "Reviewed source text: " + " ".join(CORPUS[file][page - 1][1]), ""])
    (output / "annotation_review.md").write_text("\n".join(review), encoding="utf-8")
    print(f"Created {len(documents)} original demo PDFs and {len(labels)} page labels.")


if __name__ == "__main__":
    main()
