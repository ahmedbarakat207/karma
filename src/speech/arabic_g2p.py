#!/usr/bin/env python3
"""
Arabic (MSA & Egyptian) G2P front-end for Nabra-82M / Kokoro pipeline.
======================================================================
Handles:
1. Arabic character and script detection.
2. Latin loanword normalization and whitespace tidying.
3. Optional diacritization (via camel-tools if installed, or fallback to espeak).
4. Espeak-ng G2P mapping with Kokoro Arabic phoneme preservation (ع -> 7, ح -> 8).
5. Syllable dot and unwanted bracket stripping.
"""

import re
from typing import Tuple

# Arabic Unicode range check (Arabic, Supplement, Extended-A)
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

# Espeak phoneme cleanup: out-of-vocab markers mapped to in-vocab Kokoro symbols
ARABIC_PHONEME_FIXUPS = {
    "̪": "",  # combining bridge below (dental)
    "ˤ": "",  # pharyngealization marker
    "[": "",  # espeak untranslatable bracket
    "]": "",
    "{": "",
    "}": "",
}

# Kokoro embedding slots reserved for Arabic pharyngeals
EXTRA_SYMBOLS = {
    "ʕ": 7,  # ع (voiced pharyngeal fricative)
    "ħ": 8,  # ح (voiceless pharyngeal fricative)
}

# Regex to strip mid-word syllable dots injected by espeak (e.g. ʔarrˈa.ʤul)
_SYLLABLE_DOT = re.compile(r"(?<=\S)\.(?=\S)")
_LATIN_RUN = re.compile(r"[A-Za-z]+")
_CITATION = re.compile(r"\[[^\]]*\]")
_WS = re.compile(r"\s+")


def is_arabic(text: str) -> bool:
    """Check if the text contains Arabic script characters."""
    if not text:
        return False
    return bool(_ARABIC_RE.search(text))


def normalize_arabic_text(text: str) -> Tuple[str, int]:
    """Clean citations and whitespace. Preserves Arabic script and punctuation."""
    if not text:
        return "", 0
    text = _CITATION.sub(" ", text)
    latin = _LATIN_RUN.findall(text)
    if latin:
        text = _LATIN_RUN.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text, len(latin)


def clean_phonemes(ph: str) -> str:
    """Strip syllable dots and remap out-of-vocab phonemes."""
    if not ph:
        return ""
    ph = _SYLLABLE_DOT.sub("", ph)
    for old, new in ARABIC_PHONEME_FIXUPS.items():
        ph = ph.replace(old, new)
    return ph


class ArabicG2P:
    """Text to IPA phonemes for Arabic with optional diacritization."""

    def __init__(self, diacritize: bool = True):
        self._diac_enabled = diacritize
        self._mle = None
        self.diac_available = False
        self._g2p = None

        try:
            from misaki import espeak
            self._g2p = espeak.EspeakG2P(language="ar")
        except Exception:
            self._g2p = None

        if diacritize:
            self._init_diacritizer()

    def _init_diacritizer(self):
        try:
            from camel_tools.disambig.mle import MLEDisambiguator
            self._mle = MLEDisambiguator.pretrained()
            self.diac_available = True
        except Exception:
            self._mle = None
            self.diac_available = False

    def diacritize(self, text: str) -> str:
        """Restore short vowels (tashkeel). Returns original if diacritizer unavailable."""
        if not self._mle or not text:
            return text
        try:
            from camel_tools.tokenizers.word import simple_word_tokenize
            tokens = simple_word_tokenize(text)
            if not tokens:
                return text
            out = []
            for d in self._mle.disambiguate(tokens):
                if d.analyses:
                    out.append(d.analyses[0].analysis.get("diac", d.word))
                else:
                    out.append(d.word)
            return " ".join(out)
        except Exception:
            return text

    def process(self, raw_text: str) -> Tuple[str, str]:
        """Returns (normalized_text, phonemes)."""
        text, _ = normalize_arabic_text(raw_text)
        diac = self.diacritize(text) if self._diac_enabled else text
        if self._g2p is not None:
            res = self._g2p(diac)
            ph = res[0] if isinstance(res, (tuple, list)) else str(res)
            return text, clean_phonemes(ph)
        return text, ""

    def __call__(self, raw_text: str) -> str:
        return self.process(raw_text)[1]
