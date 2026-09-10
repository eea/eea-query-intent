from __future__ import annotations

# EEA language scope: the 24 official EU languages plus Icelandic, Turkish,
# and Norwegian. Norwegian is evaluated as both written standards rather than
# collapsed into the generic `no` locale.
SUPPORTED_LANGUAGE_CODES = frozenset(
    {
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "ga",
        "hr",
        "hu",
        "is",
        "it",
        "lt",
        "lv",
        "mt",
        "nb",
        "nl",
        "nn",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "sv",
        "tr",
    }
)
