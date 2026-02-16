import json
import os
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

Difficulty = Literal["beginner", "intermediate", "advanced"]


@lru_cache(maxsize=64)
def _load_entries(language: str) -> list[dict]:
    path = CLEAN_DIR / language / "word_list_clean.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Önce scripts/build_wordlists.py çalıştır."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@tool
def get_n_random_words(language: str, n: int = 10) -> list[str]:
    """Return n random words from the cleaned word list for the given language."""
    entries = _load_entries(language)
    words = [e["word"] for e in entries]
    n = min(n, len(words))
    return random.sample(words, n)


@tool
def get_n_random_words_by_difficulty_level(
    language: str,
    word_difficulty: Difficulty,
    n: int = 10,
) -> list[str]:
    """
    Return n random words from the cleaned list filtered by difficulty.
    word_difficulty must be one of: beginner, intermediate, advanced.
    """
    entries = _load_entries(language)
    filtered = [e["word"] for e in entries if e.get("word_difficulty") == word_difficulty]
    if not filtered:
        raise ValueError(f"No words for {language=} {word_difficulty=}.")
    n = min(n, len(filtered))
    return random.sample(filtered, n)


def _translation_llm():
    provider = os.getenv("TRANSLATION_PROVIDER", "ollama").lower()
    if provider == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini"), temperature=0)
    # dokümandaki gibi: ayrı bir modelle (örn llama 3.2) çeviri :contentReference[oaicite:26]{index=26}
    return ChatOllama(model=os.getenv("OLLAMA_TRANSLATION_MODEL", "llama3.2"), temperature=0)


@tool
def translate_words(
    words: list[str],
    source_language: str,
    target_language: str,
) -> dict:
    """
    Translate words from source_language to target_language.
    Returns a JSON dict mapping each source word to its translation.
    """
    llm = _translation_llm()
    prompt = (
        f"Translate the following {source_language} words into {target_language}.\n"
        f"Return ONLY a valid JSON object mapping each source word to its translation.\n"
        f"No extra text.\n\nWords: {words}"
    )
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    # Dokümanda: model bazen ekstra text basar; { ... } yakalayıp parse ediyoruz :contentReference[oaicite:27]{index=27}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON in model output: {text[:200]}")
    data = json.loads(m.group(0))

    # sanity-check
    missing = [w for w in words if w not in data]
    if missing:
        raise ValueError(f"Missing translations for: {missing}")
    return data


# --- TAROT CLASSIFICATION (MINIMAL ADD) ---

TAROT_CATEGORIES = [
    {
        "id": "para_finans",
        "label_tr": "Para/Finans",
        "priority_keywords": ["para", "borç", "öde", "ödeme", "alacak", "verecek", "kira", "maaş", "kredi", "icra"],
        "keywords": ["bütçe", "harcama", "gelir", "zam", "ücret", "fatura", "iade"],
    },
    {
        "id": "kariyer_is",
        "label_tr": "Kariyer/İş",
        "priority_keywords": ["işe", "iş", "mülakat", "başvuru", "teklif", "terfi", "staj"],
        "keywords": ["cv", "patron", "ofis", "ekip", "proje"],
    },
    {
        "id": "ask_iliskiler",
        "label_tr": "Aşk/İlişkiler",
        "priority_keywords": ["sevgili", "aşk", "ilişki", "ex", "eski sevgili", "barış", "evlilik"],
        "keywords": ["mesaj", "geri döner", "sadakat", "kıskançlık"],
    },
    {
        "id": "aile_ev",
        "label_tr": "Aile/Ev",
        "priority_keywords": ["aile", "anne", "baba", "kardeş", "ev", "taşın"],
        "keywords": ["yuva", "kira", "ev arkadaşı"],
    },
    {
        "id": "saglik_iyiolus",
        "label_tr": "Sağlık/İyi Oluş",
        "priority_keywords": ["sağlık", "hastalık", "tedavi", "ameliyat", "terapi", "depresyon", "anksiyete"],
        "keywords": ["uyku", "stres", "şifa", "iyileş"],
    },
    {
        "id": "egitim",
        "label_tr": "Eğitim/Sınav",
        "priority_keywords": ["sınav", "final", "büt", "ders", "okul", "üniversite"],
        "keywords": ["ödev", "proje", "kurs", "sertifika"],
    },
    {
        "id": "hukuk_resmi",
        "label_tr": "Hukuk/Resmi",
        "priority_keywords": ["dava", "mahkeme", "sözleşme", "avukat", "vize", "evrak"],
        "keywords": ["itiraz", "resmi", "imza"],
    },
    {
        "id": "sosyal_cevre",
        "label_tr": "Sosyal Çevre",
        "priority_keywords": ["arkadaş", "dost", "çevre", "topluluk"],
        "keywords": ["dedikodu", "network", "grup"],
    },
    {
        "id": "ruhsal_gelisim",
        "label_tr": "Ruhsal/Kişisel Gelişim",
        "priority_keywords": ["ruhsal", "spiritüel", "karmik", "gölge", "sezgi"],
        "keywords": ["farkındalık", "niyet", "manifest"],
    },
]

_TAROT_IDS = [c["id"] for c in TAROT_CATEGORIES]


def _norm_tr(s: str) -> str:
    s = s.strip().casefold()
    s = re.sub(r"\s+", " ", s)
    return s


def _score_tarot(question: str):
    q = _norm_tr(question)
    scores = {cid: 0 for cid in _TAROT_IDS}
    hits = {cid: [] for cid in _TAROT_IDS}

    for c in TAROT_CATEGORIES:
        cid = c["id"]
        for kw in c["priority_keywords"]:
            if kw in q:
                scores[cid] += 4
                hits[cid].append(kw)
        for kw in c["keywords"]:
            if kw in q:
                scores[cid] += 1
                hits[cid].append(kw)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked, hits


def _classifier_llm():
    provider = os.getenv("REASONING_PROVIDER", "ollama").lower()
    if provider == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_REASONING_MODEL", "gpt-4o-mini"), temperature=0)
    return ChatOllama(model=os.getenv("OLLAMA_REASONING_MODEL", "qwen3"), temperature=0)


@tool
def classify_tarot_question(question: str) -> dict:
    """
    Tarot uygulaması için Türkçe soruyu kategorilere ayırır.
    primary: ana sonuç/stake (para mı iş mi ilişki mi?)
    secondary: bağlam (örn: 'eski sevgili' ilişki bağlamıdır ama para stake olabilir)
    """
    ranked, hits = _score_tarot(question)
    (best_id, best_score), (second_id, second_score) = ranked[0], ranked[1]

    if best_score >= 6 and best_score >= second_score + 3:
        secondary = [second_id] if second_score >= 3 else []
        return {
            "primary": best_id,
            "secondary": secondary,
            "confidence": 0.82,
            "signals": {k: v for k, v in hits.items() if v},
            "needs_followup": False,
            "followup": None,
            "reason_tr": "Ana kategori, sorunun asıl sonucu/stake'ine göre seçildi.",
        }

    llm = _classifier_llm()
    taxonomy = "\n".join([f'- {c["id"]}: {c["label_tr"]}' for c in TAROT_CATEGORIES])

    prompt = f"""
Görev: Tarot sorusunu sınıflandır.
Kural: Primary kategori, sorunun 'asıl sonucu/stake'i olmalı. Kişi/rol kelimeleri (eski sevgili, patron) tek başına primary yapmaz.

Kategori listesi:
{taxonomy}

Soru:
{question}

Sadece JSON döndür:
{{
  "primary": "<id>",
  "secondary": ["<id>", "..."] ,
  "confidence": 0.0,
  "reason_tr": "tek cümle"
}}
""".strip()

    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(m.group(0)) if m else {"primary": best_id, "secondary": [second_id], "confidence": 0.55, "reason_tr": "Heuristik + belirsiz."}

    if data.get("primary") not in _TAROT_IDS:
        data["primary"] = best_id
    sec = data.get("secondary") or []
    data["secondary"] = [s for s in sec if s in _TAROT_IDS and s != data["primary"]][:2]
    data["confidence"] = float(data.get("confidence", 0.55))
    data["signals"] = {k: v for k, v in hits.items() if v}

    data["needs_followup"] = data["confidence"] < 0.55
    data["followup"] = "Bunu en çok hangi sonuç için soruyorsun: para mı iş mi ilişki mi?" if data["needs_followup"] else None
    return data
