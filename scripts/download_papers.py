"""Optional public research corpus. PDFs stay in ignored data/, never committed."""
from pathlib import Path
import requests

PAPERS = {"attention.pdf": "https://arxiv.org/pdf/1706.03762",
          "bert.pdf": "https://arxiv.org/pdf/1810.04805",
          "rag.pdf": "https://arxiv.org/pdf/2005.11401"}

if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data"
    output.mkdir(exist_ok=True)
    for name, url in PAPERS.items():
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": "DocuSense-local-demo/1.0"})
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError("Response was not a PDF")
            (output / name).write_bytes(response.content)
            print(f"Downloaded {name}: {len(response.content):,} bytes from {url}")
        except (requests.RequestException, ValueError) as exc:
            print(f"Could not download {name}: {exc}. Manual PDF upload remains available.")
