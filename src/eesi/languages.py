"""Language codes and names.

A character or a player's session can be pinned to one language. The SDK then
tells Nur Live both the code (so the greeting and anything said before the
player's first words come out in that language) and a name (the language the
replies are held to). A regional code names the variety: ``ar-EG`` is
Egyptian Arabic. Anything else you pass is used as the name as written, so
``"Kansai Japanese"`` with ``language_name`` works for varieties not listed.
"""

from __future__ import annotations

__all__ = ["LANGUAGE_NAMES", "describe"]

LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "ga": "Irish",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kyrgyz",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "xh": "Xhosa",
    "yo": "Yoruba",
    "zh": "Mandarin Chinese",
    "zu": "Zulu",
    # Varieties people ask for by region.
    "ar-eg": "Egyptian Arabic",
    "ar-sa": "Saudi Arabic",
    "ar-ae": "Emirati Arabic",
    "ar-lb": "Lebanese Arabic",
    "ar-ps": "Palestinian Arabic",
    "ar-jo": "Jordanian Arabic",
    "ar-sy": "Syrian Arabic",
    "ar-iq": "Iraqi Arabic",
    "ar-ma": "Moroccan Arabic",
    "ar-dz": "Algerian Arabic",
    "ar-tn": "Tunisian Arabic",
    "ar-kw": "Kuwaiti Arabic",
    "en-us": "American English",
    "en-gb": "British English",
    "en-au": "Australian English",
    "en-in": "Indian English",
    "es-es": "Castilian Spanish",
    "es-mx": "Mexican Spanish",
    "es-ar": "Argentine Spanish",
    "pt-br": "Brazilian Portuguese",
    "pt-pt": "European Portuguese",
    "fr-ca": "Canadian French",
    "fr-fr": "French",
    "de-ch": "Swiss German",
    "de-at": "Austrian German",
    "zh-cn": "Mandarin Chinese",
    "zh-tw": "Taiwanese Mandarin",
    "zh-hk": "Cantonese",
    "yue": "Cantonese",
}


def describe(language: str, name: str | None = None) -> tuple[str | None, str]:
    """``(code, name)`` for *language*: a code (``"ar"``, ``"ar-EG"``) or a name.

    The code is the base language the speech recognizer takes (``"ar"`` for
    ``"ar-EG"``), or None when *language* is a name this module does not know.
    *name* overrides the name the model is told.
    """
    value = language.strip()
    key = value.lower().replace("_", "-")
    base = key.split("-")[0]
    if key in LANGUAGE_NAMES:
        return base, name or LANGUAGE_NAMES[key]
    if "-" in key and base in LANGUAGE_NAMES:  # a region this list does not name
        return base, name or LANGUAGE_NAMES[base]
    for code, known in LANGUAGE_NAMES.items():
        if known.lower() == key:
            return code.split("-")[0], name or known
    return None, name or value
