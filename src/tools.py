import json
import os
import random
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

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


# --- POST-VISION READING ASSEMBLER ---

CARD_OK_THRESHOLD = 0.80
CARD_WARN_THRESHOLD = 0.65
CARD_FAIL_THRESHOLD = 0.65
CLARIFICATION_BLOCKED_FLAGS = {"self_harm", "violence"}

ROMAN_NUMERALS = [
    "0",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
    "XXI",
]

MAJOR_ARCANA = [
    (0, "The Fool", "fool", ["deli"]),
    (1, "The Magician", "magician", ["buyucu", "buyu ustasi"]),
    (2, "The High Priestess", "high_priestess", ["azize", "bas rahibe"]),
    (3, "The Empress", "empress", ["imparatorice"]),
    (4, "The Emperor", "emperor", ["imparator"]),
    (5, "The Hierophant", "hierophant", ["basrahip"]),
    (6, "The Lovers", "lovers", ["asiklar"]),
    (7, "The Chariot", "chariot", ["savas arabasi"]),
    (8, "Strength", "strength", ["guc"]),
    (9, "The Hermit", "hermit", ["ermit", "inzivaci"]),
    (10, "Wheel of Fortune", "wheel_of_fortune", ["kader carki"]),
    (11, "Justice", "justice", ["adalet"]),
    (12, "The Hanged Man", "hanged_man", ["asilan adam"]),
    (13, "Death", "death", ["olum"]),
    (14, "Temperance", "temperance", ["denge"]),
    (15, "The Devil", "devil", ["seytan"]),
    (16, "The Tower", "tower", ["kule"]),
    (17, "The Star", "star", ["yildiz"]),
    (18, "The Moon", "moon", ["ay"]),
    (19, "The Sun", "sun", ["gunes"]),
    (20, "Judgement", "judgement", ["yargi", "mahkeme"]),
    (21, "The World", "world", ["dunya"]),
]

MINOR_SUIT_DEFS = {
    "cups": {
        "display": "Cups",
        "aliases": ["cups", "cup", "kupa", "kupalar"],
    },
    "wands": {
        "display": "Wands",
        "aliases": ["wands", "wand", "asalar", "asa", "degnek", "değnek"],
    },
    "swords": {
        "display": "Swords",
        "aliases": ["swords", "sword", "kilic", "kılıç", "kiliclar", "kılıçlar"],
    },
    "pentacles": {
        "display": "Pentacles",
        "aliases": ["pentacles", "pentacle", "coin", "coins", "disk", "disks", "pentakl", "tilsim", "tılsım"],
    },
}

MINOR_NUMBER_RANKS = [
    (1, "Ace", ["ace", "as", "1", "bir", "birli", "birlisi"]),
    (2, "Two", ["two", "2", "iki", "ikili", "ikilisi"]),
    (3, "Three", ["three", "3", "uc", "üç", "uclu", "üçlü", "uclusu", "üçlüsü"]),
    (4, "Four", ["four", "4", "dort", "dört", "dortlu", "dörtlü", "dortlusu", "dörtlüsü"]),
    (5, "Five", ["five", "5", "bes", "beş", "besli", "beşli", "beslisi", "beşlisi"]),
    (6, "Six", ["six", "6", "alti", "altı", "altili", "altılı", "altilisi", "altılısı"]),
    (7, "Seven", ["seven", "7", "yedi", "yedili", "yedilisi"]),
    (8, "Eight", ["eight", "8", "sekiz", "sekizli", "sekizlisi"]),
    (9, "Nine", ["nine", "9", "dokuz", "dokuzlu", "dokuzlusu"]),
    (10, "Ten", ["ten", "10", "on", "onlu", "onlusu"]),
]

MINOR_COURT_RANKS = [
    ("page", "Page", ["page", "vale", "jack", "usak", "uşak"]),
    ("knight", "Knight", ["knight", "sovalye", "şovalye", "atli", "atlı"]),
    ("queen", "Queen", ["queen", "kralice", "kraliçe"]),
    ("king", "King", ["king", "kral"]),
]

ROLE_TR = {
    "past": "Gecmis/Temel",
    "present": "Simdi",
    "future": "Gelecek/Sonuc",
    "situation": "Durum",
    "action": "Eylem/Oneri",
    "outcome": "Sonuc",
    "focus": "Odak",
}

_TR_ASCII_TABLE = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
    }
)


def _ascii_fold(text: str) -> str:
    return text.translate(_TR_ASCII_TABLE)


def _norm_text(text: str) -> str:
    norm = (text or "").strip().casefold()
    norm = norm.replace("&", " and ")
    norm = re.sub(r"[^0-9a-zçğıöşü\s]", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def _alias_variants(text: str) -> set[str]:
    norm = _norm_text(text)
    if not norm:
        return set()
    folded = _ascii_fold(norm)
    return {norm, folded}


def _add_alias(alias_to_id: dict[str, str], alias: str, card_id: str) -> None:
    for variant in _alias_variants(alias):
        alias_to_id.setdefault(variant, card_id)


def _make_role(position_index: int, role: str) -> dict[str, Any]:
    return {
        "position_index": position_index,
        "role": role,
        "role_tr": ROLE_TR.get(role, f"Pozisyon {position_index + 1}"),
    }


@lru_cache(maxsize=1)
def _build_tarot_catalog() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    registry: dict[str, dict[str, Any]] = {}
    alias_to_id: dict[str, str] = {}

    for idx, name, slug, tr_aliases in MAJOR_ARCANA:
        card_id = f"major_{idx:02d}_{slug}"
        metadata = {
            "card_id": card_id,
            "display_name": name,
            "arcana_type": "major",
            "suit": None,
            "number": idx,
            "slug": slug,
        }
        registry[card_id] = metadata

        short_name = name[4:] if name.lower().startswith("the ") else name
        aliases = {
            name,
            short_name,
            slug.replace("_", " "),
            f"{ROMAN_NUMERALS[idx]} {name}",
            f"{ROMAN_NUMERALS[idx]} {short_name}",
            f"{idx} {name}",
            f"{idx} {short_name}",
            f"{idx:02d} {short_name}",
        }
        aliases.update(tr_aliases)
        for alias in aliases:
            _add_alias(alias_to_id, alias, card_id)

    for suit, suit_data in MINOR_SUIT_DEFS.items():
        suit_display = suit_data["display"]
        suit_aliases = suit_data["aliases"]

        for number, rank_display, rank_aliases in MINOR_NUMBER_RANKS:
            card_id = f"{suit}_{number:02d}"
            display_name = f"{rank_display} of {suit_display}"
            metadata = {
                "card_id": card_id,
                "display_name": display_name,
                "arcana_type": "minor",
                "suit": suit,
                "number": number,
                "slug": f"{rank_display.lower()}_{suit}",
            }
            registry[card_id] = metadata
            for suit_alias in suit_aliases:
                for rank_alias in rank_aliases:
                    _add_alias(alias_to_id, f"{rank_alias} of {suit_alias}", card_id)
                    _add_alias(alias_to_id, f"{suit_alias} {rank_alias}", card_id)
                    _add_alias(alias_to_id, f"{rank_alias} {suit_alias}", card_id)

        for rank_code, rank_display, rank_aliases in MINOR_COURT_RANKS:
            card_id = f"{suit}_{rank_code}"
            display_name = f"{rank_display} of {suit_display}"
            metadata = {
                "card_id": card_id,
                "display_name": display_name,
                "arcana_type": "minor",
                "suit": suit,
                "number": rank_code,
                "slug": f"{rank_code}_{suit}",
            }
            registry[card_id] = metadata
            for suit_alias in suit_aliases:
                for rank_alias in rank_aliases:
                    _add_alias(alias_to_id, f"{rank_alias} of {suit_alias}", card_id)
                    _add_alias(alias_to_id, f"{suit_alias} {rank_alias}", card_id)
                    _add_alias(alias_to_id, f"{rank_alias} {suit_alias}", card_id)

    return registry, alias_to_id


@lru_cache(maxsize=1)
def _build_minor_parse_maps() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    suit_alias_map: list[tuple[str, str]] = []
    rank_alias_map: list[tuple[str, str]] = []

    for suit, data in MINOR_SUIT_DEFS.items():
        for alias in data["aliases"]:
            suit_alias_map.append((_ascii_fold(_norm_text(alias)), suit))

    for number, _, aliases in MINOR_NUMBER_RANKS:
        for alias in aliases:
            rank_alias_map.append((_ascii_fold(_norm_text(alias)), f"{number:02d}"))
    for rank_code, _, aliases in MINOR_COURT_RANKS:
        for alias in aliases:
            rank_alias_map.append((_ascii_fold(_norm_text(alias)), rank_code))

    suit_alias_map.sort(key=lambda x: len(x[0]), reverse=True)
    rank_alias_map.sort(key=lambda x: len(x[0]), reverse=True)
    return suit_alias_map, rank_alias_map


def _contains_alias(text: str, alias: str) -> bool:
    return bool(re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", text))


def _parse_minor_card_id(raw_name: str) -> str | None:
    text = _ascii_fold(_norm_text(raw_name))
    if not text:
        return None

    suit_aliases, rank_aliases = _build_minor_parse_maps()
    suit_match = None
    rank_match = None

    for alias, suit in suit_aliases:
        if _contains_alias(text, alias):
            suit_match = suit
            break
    for alias, rank_code in rank_aliases:
        if _contains_alias(text, alias):
            rank_match = rank_code
            break

    if not suit_match or not rank_match:
        return None

    if rank_match.isdigit():
        return f"{suit_match}_{rank_match}"
    return f"{suit_match}_{rank_match}"


def _parse_major_card_id(raw_name: str) -> str | None:
    text = _ascii_fold(_norm_text(raw_name))
    if not text:
        return None

    registry, _ = _build_tarot_catalog()
    candidates: list[str] = []
    for card_id, meta in registry.items():
        if meta["arcana_type"] != "major":
            continue
        slug_words = meta["slug"].split("_")
        if all(_contains_alias(text, word) for word in slug_words):
            candidates.append(card_id)

    if len(candidates) == 1:
        return candidates[0]
    return None


def _canonicalize_card(raw_name: str) -> str | None:
    registry, alias_to_id = _build_tarot_catalog()

    for variant in _alias_variants(raw_name):
        if variant in alias_to_id:
            return alias_to_id[variant]

    minor_id = _parse_minor_card_id(raw_name)
    if minor_id in registry:
        return minor_id

    major_id = _parse_major_card_id(raw_name)
    if major_id in registry:
        return major_id

    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "evet"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _normalize_flags(raw_flags: Any) -> list[str]:
    if not isinstance(raw_flags, list):
        return []
    out = []
    for flag in raw_flags:
        txt = _ascii_fold(_norm_text(str(flag)))
        if txt:
            out.append(txt)
    return sorted(set(out))


def _resolve_spread(spread_hint: Any, n_cards: int, domain: str) -> tuple[dict[str, Any], list[str], bool]:
    hint = _ascii_fold(_norm_text(str(spread_hint or "")))
    domain_norm = _ascii_fold(_norm_text(domain))
    notes: list[str] = []
    spread_low_confidence = False

    if n_cards == 1:
        spread = {
            "spread_type": "single_card",
            "roles": [_make_role(0, "focus")],
            "source": "inferred",
        }
        return spread, notes, spread_low_confidence

    if n_cards == 3:
        explicit_ppf = any(token in hint for token in ["past_present_future", "past present future", "gecmis simdi gelecek"])
        explicit_sao = any(token in hint for token in ["situation_action_outcome", "situation action outcome", "durum eylem sonuc"])

        if explicit_ppf:
            spread_type = "past_present_future"
            source = "hint"
        elif explicit_sao:
            spread_type = "situation_action_outcome"
            source = "hint"
        else:
            is_career_finance = any(
                token in domain_norm for token in ["career", "kariyer", "is", "finance", "finans", "para"]
            )
            spread_type = "situation_action_outcome" if is_career_finance else "past_present_future"
            source = "inferred"

        if spread_type == "past_present_future":
            roles = [
                _make_role(0, "past"),
                _make_role(1, "present"),
                _make_role(2, "future"),
            ]
        else:
            roles = [
                _make_role(0, "situation"),
                _make_role(1, "action"),
                _make_role(2, "outcome"),
            ]
        spread = {
            "spread_type": spread_type,
            "roles": roles,
            "source": source,
        }
        return spread, notes, spread_low_confidence

    generic_roles = [
        {
            "position_index": idx,
            "role": f"position_{idx + 1}",
            "role_tr": f"Pozisyon {idx + 1}",
        }
        for idx in range(max(0, n_cards))
    ]
    spread = {
        "spread_type": f"{n_cards}_card_generic" if n_cards > 0 else "unknown",
        "roles": generic_roles,
        "source": "inferred",
    }
    if n_cards > 0:
        notes.append("spread_inferred_low_confidence")
        spread_low_confidence = True
    return spread, notes, spread_low_confidence


def _build_clarification(
    vision_quality: str,
    unknown_count: int,
    n_cards: int,
    low_conf_count: int,
    spread_low_confidence: bool,
    flags: list[str],
    app_mode: str,
) -> tuple[bool, str | None, list[str], bool]:
    blocked_by_flags = any(flag in CLARIFICATION_BLOCKED_FLAGS for flag in flags)
    no_followups_mode = app_mode in {"no_followups", "no_follow_ups"}
    allow = not blocked_by_flags and not no_followups_mode

    trigger = (
        vision_quality == "fail"
        or (unknown_count >= 1 and n_cards <= 3)
        or (low_conf_count >= 2)
        or spread_low_confidence
    )
    if not trigger:
        return False, None, [], False
    if not allow:
        return False, None, [], True

    if vision_quality == "fail":
        return (
            True,
            "Kart tespiti guvenilir degil; fotoğrafi tekrar yuklemek ister misin?",
            ["Tekrar yukle", "Mevcut tespitle devam et"],
            False,
        )
    if unknown_count >= 1 and n_cards <= 3:
        return (
            True,
            "Bazi kart adlari belirsiz; kart adlarini tekrar dogrular misin?",
            ["Kart adlarini duzelt", "Belirsiz kartlarla devam et"],
            False,
        )
    if spread_low_confidence and n_cards == 3:
        return (
            True,
            "Bu dizilim icin hangi yayilim kullanilsin?",
            ["Gecmis-Simdi-Gelecek", "Durum-Eylem-Sonuc"],
            False,
        )
    return (
        True,
        "En az iki kartin guveni dusuk; yeniden cekim yapmak ister misin?",
        ["Yeniden cekim yap", "Mevcut tespitle devam et"],
        False,
    )


def _build_question_block(router_input: dict[str, Any], language: str, merged_flags: list[str]) -> dict[str, Any]:
    return {
        "text": str(router_input.get("question_text") or "").strip(),
        "language": language,
        "domain": str(router_input.get("domain") or "general").strip() or "general",
        "sub_intent": str(router_input.get("sub_intent") or "unknown").strip() or "unknown",
        "time_horizon": str(router_input.get("time_horizon") or "unknown").strip() or "unknown",
        "sensitivity_flags": merged_flags,
        "needs_disclaimer": bool(merged_flags) or bool(router_input.get("needs_disclaimer", False)),
    }


def _empty_quality() -> dict[str, Any]:
    return {
        "vision_quality": "fail",
        "uncertain_cards": [],
        "clarification_needed": False,
        "clarify_question": None,
        "clarify_choices": [],
    }


@tool
def assemble_post_vision_reading(
    vision_input: dict[str, Any],
    router_input: dict[str, Any],
    user_preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a normalized post-vision payload from vision + router outputs.
    This tool is deterministic and returns a single downstream-safe JSON.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    notes: list[str] = []
    uncertain_cards: list[dict[str, Any]] = []
    user_preferences = user_preferences if isinstance(user_preferences, dict) else {}

    if not isinstance(vision_input, dict):
        vision_input = {}
        notes.append("invalid_vision_input")
    if not isinstance(router_input, dict):
        router_input = {}
        notes.append("invalid_router_input")

    language = str(vision_input.get("language") or "tr").strip() or "tr"
    router_flags = _normalize_flags(router_input.get("sensitivity_flags"))
    vision_flags: list[str] = []
    merged_flags = sorted(set(router_flags + vision_flags))

    app_mode = _ascii_fold(_norm_text(str(user_preferences.get("app_mode") or "")))
    question = _build_question_block(router_input, language, merged_flags)

    cards_raw = vision_input.get("cards")
    quality = _empty_quality()
    spread, spread_notes, spread_low_confidence = _resolve_spread(
        vision_input.get("spread_hint"),
        0,
        question["domain"],
    )
    notes.extend(spread_notes)

    if not isinstance(cards_raw, list) or not cards_raw:
        notes.append("missing_or_empty_cards")
        clarify_needed, clarify_q, clarify_choices, clarify_suppressed = _build_clarification(
            vision_quality="fail",
            unknown_count=0,
            n_cards=0,
            low_conf_count=0,
            spread_low_confidence=False,
            flags=merged_flags,
            app_mode=app_mode,
        )
        if clarify_suppressed:
            notes.append("clarification_suppressed_due_to_policy")

        quality.update(
            {
                "clarification_needed": clarify_needed,
                "clarify_question": clarify_q,
                "clarify_choices": clarify_choices,
            }
        )
        return {
            "question": question,
            "spread": spread,
            "cards": [],
            "quality": quality,
            "meta": {
                "pipeline_version": "v1",
                "notes": sorted(set(notes)),
                "timestamps": {"assembled_at": now_iso},
                "preferences": user_preferences,
                "metrics": {
                    "vision_quality_ok": 0,
                    "vision_quality_warn": 0,
                    "vision_quality_fail": 1,
                    "clarifications_triggered": int(clarify_needed),
                    "unknown_card_rate": 0.0,
                    "avg_card_confidence": 0.0,
                },
            },
        }

    parsed_cards: list[dict[str, Any]] = []
    hard_fail = False
    for idx, item in enumerate(cards_raw):
        if not isinstance(item, dict):
            notes.append(f"invalid_card_payload_{idx}")
            hard_fail = True
            continue

        raw_name = str(item.get("name") or "").strip()
        if not raw_name:
            raw_name = "unknown"
            notes.append(f"missing_card_name_{idx}")

        try:
            position_index = int(item.get("position_index"))
        except (TypeError, ValueError):
            notes.append(f"invalid_position_index_{idx}")
            hard_fail = True
            continue

        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            notes.append(f"invalid_confidence_{idx}")
            hard_fail = True
            continue

        if confidence < 0 or confidence > 1:
            notes.append(f"out_of_range_confidence_{idx}")
            hard_fail = True
            continue

        parsed_cards.append(
            {
                "position_index": position_index,
                "raw_name": raw_name,
                "reversed": _to_bool(item.get("reversed", False)),
                "confidence": confidence,
                "warnings": [],
            }
        )

    if hard_fail or not parsed_cards:
        spread, spread_notes, spread_low_confidence = _resolve_spread(
            vision_input.get("spread_hint"),
            0,
            question["domain"],
        )
        notes.extend(spread_notes)

        clarify_needed, clarify_q, clarify_choices, clarify_suppressed = _build_clarification(
            vision_quality="fail",
            unknown_count=0,
            n_cards=0,
            low_conf_count=0,
            spread_low_confidence=spread_low_confidence,
            flags=merged_flags,
            app_mode=app_mode,
        )
        if clarify_suppressed:
            notes.append("clarification_suppressed_due_to_policy")
        quality.update(
            {
                "clarification_needed": clarify_needed,
                "clarify_question": clarify_q,
                "clarify_choices": clarify_choices,
            }
        )

        return {
            "question": question,
            "spread": spread,
            "cards": [],
            "quality": quality,
            "meta": {
                "pipeline_version": "v1",
                "notes": sorted(set(notes)),
                "timestamps": {"assembled_at": now_iso},
                "preferences": user_preferences,
                "metrics": {
                    "vision_quality_ok": 0,
                    "vision_quality_warn": 0,
                    "vision_quality_fail": 1,
                    "clarifications_triggered": int(clarify_needed),
                    "unknown_card_rate": 0.0,
                    "avg_card_confidence": 0.0,
                },
            },
        }

    by_position: dict[int, list[dict[str, Any]]] = {}
    for card in parsed_cards:
        by_position.setdefault(card["position_index"], []).append(card)

    deduped_cards: list[dict[str, Any]] = []
    for position_index in sorted(by_position):
        bucket = sorted(by_position[position_index], key=lambda c: c["confidence"], reverse=True)
        keep = bucket[0]
        if len(bucket) > 1:
            keep["warnings"].append("duplicate_position")
            notes.append("duplicate_positions_repaired")
            for dropped in bucket[1:]:
                uncertain_cards.append(
                    {
                        "position_index": position_index,
                        "reason": "duplicate_position",
                        "confidence": dropped["confidence"],
                    }
                )
        deduped_cards.append(keep)

    deduped_cards.sort(key=lambda c: c["position_index"])
    actual_indices = [c["position_index"] for c in deduped_cards]
    expected_indices = list(range(len(deduped_cards)))
    if actual_indices != expected_indices:
        notes.append("renumbered_positions")
        for idx, card in enumerate(deduped_cards):
            if card["position_index"] != idx:
                card["warnings"].append("renumbered_position")
            card["position_index"] = idx

    spread, spread_notes, spread_low_confidence = _resolve_spread(
        vision_input.get("spread_hint"),
        len(deduped_cards),
        question["domain"],
    )
    notes.extend(spread_notes)

    registry, _ = _build_tarot_catalog()
    unknown_count = 0
    low_conf_count = 0
    fail_conf_count = 0
    resolved_cards: list[dict[str, Any]] = []
    confidence_sum = 0.0

    for card in deduped_cards:
        card_id = _canonicalize_card(card["raw_name"])
        confidence = card["confidence"]
        confidence_sum += confidence
        warnings = list(card["warnings"])

        if confidence < CARD_WARN_THRESHOLD:
            low_conf_count += 1
            uncertain_cards.append(
                {
                    "position_index": card["position_index"],
                    "reason": "low_confidence",
                    "confidence": confidence,
                }
            )
        if confidence < CARD_FAIL_THRESHOLD:
            fail_conf_count += 1
        if confidence < CARD_OK_THRESHOLD and confidence >= CARD_WARN_THRESHOLD:
            warnings.append("confidence_warn")

        if not card_id or card_id not in registry:
            unknown_count += 1
            warnings.append("unknown_card")
            uncertain_cards.append(
                {
                    "position_index": card["position_index"],
                    "reason": "unknown_card",
                    "confidence": confidence,
                }
            )
            resolved_cards.append(
                {
                    "position_index": card["position_index"],
                    "raw_name": card["raw_name"],
                    "card_id": "unknown",
                    "display_name": card["raw_name"],
                    "arcana_type": "unknown",
                    "suit": None,
                    "number": None,
                    "reversed": card["reversed"],
                    "confidence": confidence,
                    "warnings": sorted(set(warnings)),
                }
            )
            continue

        meta = registry[card_id]
        resolved_cards.append(
            {
                "position_index": card["position_index"],
                "raw_name": card["raw_name"],
                "card_id": card_id,
                "display_name": meta["display_name"],
                "arcana_type": meta["arcana_type"],
                "suit": meta["suit"],
                "number": meta["number"],
                "reversed": card["reversed"],
                "confidence": confidence,
                "warnings": sorted(set(warnings)),
            }
        )

    if fail_conf_count > 0:
        vision_quality = "fail"
    elif unknown_count > 0 or low_conf_count > 0 or spread_low_confidence:
        vision_quality = "warn"
    else:
        vision_quality = "ok"

    clarify_needed, clarify_q, clarify_choices, clarify_suppressed = _build_clarification(
        vision_quality=vision_quality,
        unknown_count=unknown_count,
        n_cards=len(resolved_cards),
        low_conf_count=low_conf_count,
        spread_low_confidence=spread_low_confidence,
        flags=merged_flags,
        app_mode=app_mode,
    )
    if clarify_suppressed:
        notes.append("clarification_suppressed_due_to_policy")

    quality = {
        "vision_quality": vision_quality,
        "uncertain_cards": uncertain_cards,
        "clarification_needed": clarify_needed,
        "clarify_question": clarify_q,
        "clarify_choices": clarify_choices,
    }

    avg_conf = round(confidence_sum / len(resolved_cards), 4) if resolved_cards else 0.0
    unknown_rate = round(unknown_count / len(resolved_cards), 4) if resolved_cards else 0.0

    return {
        "question": question,
        "spread": spread,
        "cards": resolved_cards,
        "quality": quality,
        "meta": {
            "pipeline_version": "v1",
            "notes": sorted(set(notes)),
            "timestamps": {"assembled_at": now_iso},
            "preferences": user_preferences,
            "metrics": {
                "vision_quality_ok": int(vision_quality == "ok"),
                "vision_quality_warn": int(vision_quality == "warn"),
                "vision_quality_fail": int(vision_quality == "fail"),
                "clarifications_triggered": int(clarify_needed),
                "unknown_card_rate": unknown_rate,
                "avg_card_confidence": avg_conf,
            },
        },
    }
