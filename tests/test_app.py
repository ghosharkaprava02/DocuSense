"""Real Streamlit script execution without a browser or network downloads."""
from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_empty_app_and_keyword_demo():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30).run()
    assert not app.exception
    app.selectbox[0].select("Keyword · TF-IDF baseline").run()
    next(b for b in app.button if b.label == "Load reproducible demo").click().run()
    assert not app.exception
    assert app.session_state["index"].mode == "keyword (TF-IDF)"
    next(b for b in app.button if b.label == "Load reviewed demo labels").click().run()
    next(b for b in app.button if b.label == "Run retrieval evaluation").click().run()
    assert not app.exception
    assert app.session_state["evaluation"] is not None
