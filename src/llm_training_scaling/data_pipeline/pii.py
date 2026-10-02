import re


EMAIL_PLACEHOLDER = "|||EMAIL_ADDRESS|||"
PHONE_PLACEHOLDER = "|||PHONE_NUMBER|||"
IP_PLACEHOLDER = "|||IP_ADDRESS|||"


# Practical email matcher for ordinary web text.
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9-]+"
    r"(?:\.[A-Za-z0-9-]+)+\b"
)

# Matches common 10-digit US formats:
# 2831823829
# 283-182-3829
# (283)-182-3829
# (283) 182 3829
_PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\(\s*\d{3}\s*\)|\d{3})"
    r"[\s.-]*"
    r"\d{3}"
    r"[\s.-]*"
    r"\d{4}"
    r"(?!\d)"
)

# Each IPv4 octet must be between 0 and 255.
_IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
_IP_PATTERN = re.compile(
    rf"(?<![\d.])"
    rf"{_IPV4_OCTET}(?:\.{_IPV4_OCTET}){{3}}"
    rf"(?!\d|\.\d)"
)


def mask_emails(text: str) -> tuple[str, int]:
    """Replace email addresses and return the replacement count."""
    return _EMAIL_PATTERN.subn(EMAIL_PLACEHOLDER, text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Replace common US phone-number formats."""
    return _PHONE_PATTERN.subn(PHONE_PLACEHOLDER, text)


def mask_ips(text: str) -> tuple[str, int]:
    """Replace valid IPv4 addresses."""
    return _IP_PATTERN.subn(IP_PLACEHOLDER, text)
