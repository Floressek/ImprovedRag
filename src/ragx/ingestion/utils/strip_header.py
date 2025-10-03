from typing import Optional
import re


def extract_section_header(text: str) -> Optional[str]:
    m = re.match(r"^(={2,6})\s*(.+?)\s*\1", text, re.MULTILINE)
    if m:
        return m.group(2).strip()
    lines = text.split("\n", 1)
    if lines and len(lines[0]) < 100 and lines[0].strip():
        first_line = lines[0].strip()
        if len(first_line.split()) <= 10:
            return first_line
    return None

def strip_leading_header(text: str) -> str:
    lines = text.split("\n")
    if not lines:
        return text
    head = lines[0]
    if re.match(r"^(={2,6})\s*(.+?)\s*\1\s*$", head):
        return "\n".join(lines[1:]).lstrip()
    return text