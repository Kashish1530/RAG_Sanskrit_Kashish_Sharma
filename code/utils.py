import re
import html

def clean_text(text: str) -> str:
    # ✅ Convert &quot; &#39; etc → real characters
    text = html.unescape(text)

    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()