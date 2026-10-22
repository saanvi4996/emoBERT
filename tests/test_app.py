"""Loads the real model (Hub download, or EMOBERT_MODEL pointing at a local copy) and runs inference."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run(text):
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    at.text_area[0].input(text)
    at.button[0].click().run()
    assert not at.exception
    return at


def test_predicts_joy():
    at = run("I just got my dream internship, I am so happy!")
    assert "JOY" in at.success[0].value


def test_predicts_anger():
    at = run("I am furious that they cancelled my flight again without telling me.")
    assert "ANGER" in at.success[0].value


def test_empty_input_warns():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.button[0].click().run()
    assert at.warning
