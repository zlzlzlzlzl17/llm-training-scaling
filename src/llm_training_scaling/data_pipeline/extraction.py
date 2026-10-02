from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import bytes_to_str, detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """Decode raw HTML bytes and extract visible plain text."""
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html = bytes_to_str(html_bytes, encoding)

    return extract_plain_text(html)
