"""Media Processing & Multimodal Helper for Images and Videos.
Menangani konversi base64 dan persiapan payload multimodal untuk Gemini Vision.
"""

import os
import base64
import mimetypes
from typing import List, Dict, Any, Optional


def encode_file_to_base64(file_path: str) -> Optional[str]:
    """Mengonversi file gambar atau video lokal ke format base64 string."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[Error] Gagal encode base64 untuk {file_path}: {e}")
        return None


def get_mime_type(file_path: str) -> str:
    """Mendapatkan MIME type dari ekstensi file."""
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        return mime
    
    ext = file_path.lower().split(".")[-1]
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "avi": "video/x-msvideo"
    }
    return mime_map.get(ext, "application/octet-stream")


def build_multimodal_content_blocks(prompt_text: str, media_paths: List[str]) -> List[Dict[str, Any]]:
    """Membangun blok pesan multimodal (Text + Image/Video Base64) untuk LangChain Gemini."""
    content_blocks: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt_text}
    ]

    for m_path in media_paths:
        if not os.path.exists(m_path):
            continue

        b64_data = encode_file_to_base64(m_path)
        if not b64_data:
            continue

        mime_type = get_mime_type(m_path)

        if mime_type.startswith("image/"):
            content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_data}"
                }
            })
        elif mime_type.startswith("video/"):
            # Format video data URL untuk Gemini
            content_blocks.append({
                "type": "media",
                "mime_type": mime_type,
                "data": b64_data
            })

    return content_blocks
