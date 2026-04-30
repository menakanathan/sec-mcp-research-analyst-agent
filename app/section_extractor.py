import re
from app.models import ExtractedSection

ITEM_PATTERNS = {
    "1": ("Business", [r"Item\s+1\.\s+Business", r"ITEM\s+1\s+BUSINESS"]),
    "1A": ("Risk Factors", [r"Item\s+1A\.\s+Risk\s+Factors", r"ITEM\s+1A\s+RISK\s+FACTORS"]),
    "7": ("Management Discussion and Analysis", [r"Item\s+7\.\s+Management", r"ITEM\s+7\s+MANAGEMENT"]),
    "8": ("Financial Statements", [r"Item\s+8\.\s+Financial\s+Statements", r"ITEM\s+8\s+FINANCIAL\s+STATEMENTS"]),
}

NEXT_ITEM_ORDER = {"1": ["1A", "1B", "2"], "1A": ["1B", "2"], "7": ["7A", "8"], "8": ["9", "9A"]}


def _normalize_item(item: str) -> str:
    """
    Normalize the item argument to a bare key like "1A", "7", "8".

    Accepts any of these formats from the LLM:
        "1A"  "1a"  "item 1a"  "ITEM 1A"  "Item 1A"  "item1a"
    Returns the uppercase bare form expected by ITEM_PATTERNS.
    """
    # Strip surrounding whitespace and convert to uppercase
    s = item.strip().upper()
    # Remove leading "ITEM" word followed by optional space/hyphen/underscore
    s = re.sub(r"^ITEM[\s\-_]*", "", s).strip()
    return s


def extract_section(text: str, item: str, max_chars: int = 25000) -> ExtractedSection:
    item = _normalize_item(item)
    title, patterns = ITEM_PATTERNS.get(item, (item, []))
    if not patterns:
        raise ValueError(f"Unsupported item: {item!r}. "
                         f"Valid values: {list(ITEM_PATTERNS)}")

    start = None
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            start = m.start()
            break

    if start is None:
        return ExtractedSection(item=item, title=title, text="", char_count=0)

    candidates = []
    search_text = text[start + 50:]
    for next_item in NEXT_ITEM_ORDER.get(item, []):
        for pat in [rf"Item\s+{re.escape(next_item)}\.", rf"ITEM\s+{re.escape(next_item)}"]:
            m = re.search(pat, search_text, flags=re.IGNORECASE)
            if m:
                candidates.append(start + 50 + m.start())

    end = min(candidates) if candidates else None
    section = text[start:end] if end else text[start:]
    section = re.sub(r"\s+", " ", section).strip()

    if len(section) > max_chars:
        section = section[:max_chars] + " ... [TRUNCATED]"

    return ExtractedSection(item=item, title=title, text=section, char_count=len(section))