"""Render maintained Markdown guides into styled, paginated PDF artifacts."""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, KeepTogether

ROOT = Path(__file__).resolve().parents[1]


def build(source: Path, output: Path):
    styles = getSampleStyleSheet()
    for name in ["Normal", "BodyText"]:
        styles[name].fontSize = 10.5
        styles[name].leading = 15
        styles[name].textColor = colors.HexColor("#21334a")
        styles[name].spaceAfter = 11
        styles[name].allowOrphans = 0
        styles[name].allowWidows = 0
    styles["Title"].textColor = colors.HexColor("#122e45")
    styles["Title"].fontSize = 32
    styles["Title"].leading = 38
    styles["Heading2"].textColor = colors.HexColor("#0b776e")
    styles["Heading2"].spaceBefore = 15
    story = []
    content = source.read_text(encoding="utf-8").replace("\n## ", "\n\n## ")
    for block in content.strip().split("\n\n"):
        block = block.strip()
        if not block.strip():
            continue
        if block.startswith("# "):
            story.extend([Paragraph(escape(block[2:]), styles["Title"]), Spacer(1, 12)])
        elif block.startswith("## "):
            story.append(Paragraph(escape(block[3:]), styles["Heading2"]))
        else:
            story.append(Paragraph(escape(block.replace("\n", " ")), styles["BodyText"]))
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#bedbd6"))
        canvas.line(50, 44, 562, 44)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#52647a"))
        canvas.drawString(50, 30, "DOCUSENSE  /  LOCAL DOCUMENT INTELLIGENCE")
        canvas.drawRightString(562, 30, str(doc.page))
    grouped = []
    i = 0
    while i < len(story):
        if isinstance(story[i], Paragraph) and story[i].style.name == "Heading2" and i + 1 < len(story):
            grouped.append(KeepTogether(story[i:i + 2]))
            i += 2
        else:
            grouped.append(story[i])
            i += 1
    SimpleDocTemplate(str(output), pagesize=(612, 792), topMargin=48, bottomMargin=62,
                      leftMargin=50, rightMargin=50, invariant=1,
                      allowSplitting=1, title=source.stem.replace("_", " ")).build(
                          grouped, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    output = ROOT / "output" / "pdf"
    output.mkdir(parents=True, exist_ok=True)
    for name in ["technical_guide", "interview_guide"]:
        path = output / f"DocuSense_{name}.pdf"
        build(ROOT / "docs" / f"{name}.md", path)
        print(path)
