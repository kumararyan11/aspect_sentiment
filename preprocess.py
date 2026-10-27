"""Custom cleaning + tokenization for noisy user-generated review text.

Handles: HTML/URLs, emojis & emoticons, slang/abbreviations, elongated words, repeated punctuation,
contractions, and common romanized-Hindi (Hinglish) snippets.
"""
import html
import re
import unicodedata

import emoji

URL = re.compile(r"https?://\S+|www\.\S+")
TAG = re.compile(r"<[^>]+>")
MENTION = re.compile(r"@\w+")
PUNCT_REP = re.compile(r"([!?.,])\1+")
ELONG = re.compile(r"([a-z])\1{2,}")
WS = re.compile(r"\s+")

EMOJI_SENTIMENT = {
    "😍": "love great", "😊": "happy", "😀": "happy", "😃": "happy", "😁": "happy", "🥰": "love great",
    "❤": "love", "👍": "good", "👌": "good", "🔥": "excellent", "💯": "perfect", "⭐": "great", "🙏": "thanks",
    "😡": "angry bad", "😠": "angry bad", "🤬": "terrible", "👎": "bad", "😞": "disappointed", "😢": "sad",
    "😭": "sad disappointed", "🤮": "terrible", "😤": "annoyed", "😐": "okay", "😑": "mediocre", "🙄": "annoyed",
}
EMOTICONS = {":)": "happy", ":-)": "happy", ":D": "happy", "<3": "love", ":(": "sad", ":-(": "sad", ":/": "unhappy", ":|": "okay"}

SLANG = {
    "gr8": "great", "awsm": "awesome", "awesum": "awesome", "pls": "please", "plz": "please", "thx": "thanks",
    "thnx": "thanks", "tnx": "thanks", "u": "you", "ur": "your", "r": "are", "nyc": "nice", "gud": "good",
    "gd": "good", "bcoz": "because", "bcz": "because", "cuz": "because", "luv": "love", "lov": "love",
    "amazin": "amazing", "ok": "okay", "nt": "not", "bt": "but", "tbh": "to be honest", "meh": "mediocre",
    "rly": "really", "qty": "quantity", "pkg": "packaging", "pkging": "packaging", "dlvry": "delivery",
    "deliv": "delivery", "vfm": "value for money", "wtf": "terrible", "omg": "wow", "lol": "funny",
    "cant": "can not", "wont": "will not", "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "isnt": "is not", "wasnt": "was not", "awsome": "awesome", "excellant": "excellent", "gud": "good",
}

HINGLISH = {
    "paisa vasool": "value for money", "paisa wasool": "value for money", "accha": "good", "acha": "good",
    "achha": "good", "bahut": "very", "bohot": "very", "bekar": "useless", "bakwas": "rubbish",
    "ghatiya": "poor quality", "kharab": "bad", "badhiya": "excellent", "zabardast": "excellent",
    "mehenga": "expensive", "mehnga": "expensive", "sasta": "cheap", "dhokha": "fraud", "kamaal": "amazing",
    "faltu": "worthless", "sahi": "good", "nahi": "not", "nahin": "not", "jaldi": "quickly",
}
_HING_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, HINGLISH), key=len, reverse=True)) + r")\b")
_SLANG_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SLANG), key=len, reverse=True)) + r")\b")


def _convert_emojis(text: str) -> str:
    for e, w in EMOJI_SENTIMENT.items():
        text = text.replace(e + "\ufe0f", f" {w} ").replace(e, f" {w} ")
    for e, w in EMOTICONS.items():
        text = text.replace(e, f" {w} ")
    text = emoji.demojize(text, delimiters=(" ", " "))  # remaining emojis -> words
    return text.replace("_", " ")


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("`", "'")
    text = URL.sub(" ", text)
    text = TAG.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = _convert_emojis(text)
    text = text.lower()
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = ELONG.sub(r"\1\1", text)                 # goooood -> good(oo)
    text = _HING_RE.sub(lambda m: HINGLISH[m.group(1)], text)
    text = _SLANG_RE.sub(lambda m: SLANG[m.group(1)], text)
    text = PUNCT_REP.sub(r"\1", text)
    return WS.sub(" ", text).strip()


NEGATORS = {"not", "no", "never", "cannot", "hardly", "without", "neither", "nor"}
_TOK = re.compile(r"[a-z0-9']+|[.,;:!?]")


def custom_tokenize(text: str) -> list[str]:
    """Regex tokenizer with scope-limited negation marking (used by the TF-IDF baseline)."""
    out, neg = [], 0
    for t in _TOK.findall(text.lower()):
        if t in ".,;:!?":
            neg = 0
            if t in "!?":
                out.append(t)
        elif t in NEGATORS:
            out.append(t)
            neg = 3
        elif neg > 0:
            out.append("NOT_" + t)
            neg -= 1
        else:
            out.append(t)
    return out


if __name__ == "__main__":
    print(clean_text("Gr8 product!!! 😍😍 but delivery was sooooo late 😡 paisa vasool tho. Pls don't buy <b>this</b> https://x.co"))
