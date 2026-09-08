"""Output Formatter & Cleaner for LLM / Multi-Agent Responses.
Membersihkan format mentah (seperti list thinking tokens, escaped JSON/dict, <think> tags,
perulangan kata looping, dan deretan karakter pembatas seperti ///////// atau ========)
dari Gemma 4 / Gemini agar tampil sebagai Markdown yang rapi dan elegan di UI.
"""

import ast
import json
import re
from typing import Any, Tuple, Optional


def clean_and_format_output(raw_content: Any) -> str:
    """Mengubah format output mentah dari LLM menjadi teks Markdown bersih serta memotong loop kata dan karakter."""
    if not raw_content:
        return ""

    # 1. Jika raw_content bertipe list (e.g. [{'type': 'thinking', ...}, {'type': 'text', 'text': '...'}])
    if isinstance(raw_content, list):
        text_parts = []
        for item in raw_content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(str(item["text"]))
                elif "content" in item:
                    text_parts.append(str(item["content"]))
                elif "text" in item and item.get("type") != "thinking":
                    text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
        if text_parts:
            text = "\n\n".join(text_parts).strip()
        else:
            text = "\n\n".join(str(x) for x in raw_content).strip()
    else:
        text = str(raw_content).strip()

    # 2. Cek apakah string merupakan stringified Python list atau JSON (seperti pada Gemma 4)
    if (text.startswith("[{") and text.endswith("}]")) or (text.startswith("({'") and text.endswith("})")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                extracted_texts = []
                for elem in parsed:
                    if isinstance(elem, dict):
                        if elem.get("type") == "text" and "text" in elem:
                            extracted_texts.append(elem["text"])
                        elif "text" in elem and elem.get("type") != "thinking":
                            extracted_texts.append(elem["text"])
                        elif "content" in elem:
                            extracted_texts.append(elem["content"])
                if extracted_texts:
                    text = "\n\n".join(extracted_texts).strip()
        except Exception:
            matches = re.findall(r"'(?:text|content)'\s*:\s*['\"](.*?)['\"](?:\s*[,}])", text, re.DOTALL)
            if matches:
                text = "\n\n".join(matches).strip()

    # 3. Bersihkan tag <think>...</think> atau <thought>...</thought> jika ada
    text = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>", "", text, flags=re.DOTALL).strip()

    # 4. Perbaiki karakter escape seperti literal \n, \t, dan backslash ganda
    if "\\n" in text:
        text = text.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'")

    # 5. ANTI-DEGENERATIVE CHARACTER LOOP FILTER:
    # Memotong deretan simbol berulang tanpa jeda seperti #///////////////// atau ============= atau -----------
    text = re.sub(r'[\/\\#=\-_*~]{4,}', '', text)

    # 6. ANTI-DEGENERATIVE TOKEN REPETITION FILTER:
    # Memotong perulangan kata berulang kali (seperti 'owns owns owns owns...')
    text = re.sub(r'(\b\w+\b)(?:\s+\1){3,}', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'(\S+)(?:\s+\1){3,}', r'\1', text)

    # 7. Hapus baris yang berulang-ulang
    lines = text.split('\n')
    deduped_lines = []
    repeat_count = 0
    last_line = None
    for line in lines:
        stripped = line.strip()
        if stripped and stripped == last_line:
            repeat_count += 1
            if repeat_count < 2:
                deduped_lines.append(line)
        else:
            repeat_count = 0
            last_line = stripped
            deduped_lines.append(line)
    text = '\n'.join(deduped_lines)

    # 8. Rapikan spasi berlebih
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_python_code(text: str) -> str:
    """Mengekstrak seluruh blok kode Python dari teks respon LLM dan menggabungkannya secara runtut."""
    if not text:
        return ""
    matches = re.findall(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return "\n\n# --- Kode Eksekusi Multi-Agent ---\n".join(m.strip() for m in matches if m.strip())
    return ""
