"""Guard the two CSS contracts; actual rendering is checked in the browser."""
from html.parser import HTMLParser
from pathlib import Path
import re

STATIC = Path(__file__).resolve().parent.parent / "nucleo_viewer" / "static"


class ImageAttributes(HTMLParser):
    def __init__(self):
        super().__init__()
        self.attributes = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "img" and attributes.get("id") == "ct-image":
            self.attributes = attributes


def css_rule(selector):
    css = (STATIC / "styles.css").read_text()
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert match is not None, f"Missing layout rule: {selector}"
    return dict(part.strip().split(":", 1) for part in match[1].split(";") if ":" in part)


def test_empty_viewer_hides_image_even_with_explicit_image_display_rule():
    parser = ImageAttributes()
    parser.feed((STATIC / "index.html").read_text())
    assert parser.attributes is not None
    assert "hidden" in parser.attributes
    assert "src" not in parser.attributes
    assert css_rule(".viewport img[hidden]")["display"].strip() == "none"


def test_long_study_id_can_shrink_and_wrap_inside_sidebar():
    rule = css_rule(".case-details dd")
    assert rule["min-width"].strip() == "0"
    assert rule["overflow-wrap"].strip() == "anywhere"
    assert css_rule(".control-panel")["min-width"].strip() == "0"
