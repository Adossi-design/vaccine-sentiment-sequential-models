"""This module contains the text cleaning shared by all five models, which turns a raw tweet into space-separated tokens."""
import html
import re
from dataclasses import asdict, dataclass

# These Unicode ranges cover the emojis found in the tweets so that each emoji becomes its own token.
EMOJI_CLASS = "\U0001F300-\U0001FAFF☀-➿"
EMOJI_RE = re.compile(f"[{EMOJI_CLASS}]")

_TOKEN_RE = re.compile(
    r"<url>|<user>"              # Link and username placeholders are kept as single tokens.
    r"|#\w+"                     # Hashtags are matched whole and later handled by TextConfig.hashtags.
    r"|\w+(?:'\w+)?"             # Words are matched with contractions such as isn't kept together.
    "|[!?…]"                # Exclamation marks, question marks and ellipses are kept as tokens.
    f"|[{EMOJI_CLASS}]"
)
_ELONGATION_RE = re.compile(r"([^\W\d_])\1{2,}")   # Only repeated letters are shortened so "noooooo" becomes "noo" while numbers stay unchanged.
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


@dataclass(frozen=True)
class TextConfig:
    """This class holds the text cleaning settings, whose default values were chosen from the EDA."""

    lowercase: bool = True           # Lowercasing makes "Vaccine", "VACCINE" and "vaccine" the same token.
    hashtags: str = "keep"           # The value "keep" produces '#vaccineswork' and the value "strip" produces 'vaccineswork'.
    keep_punct: bool = True          # This setting keeps exclamation marks, question marks and ellipses as tokens.
    keep_emoji: bool = True
    squeeze_elongation: bool = True

    def describe(self):
        """Return the settings as one line of text for printing."""
        return ", ".join(f"{name}={value}" for name, value in asdict(self).items())


DEFAULT_CONFIG = TextConfig()


def tokenize(text, config=DEFAULT_CONFIG):
    """Split one tweet into a list of cleaned tokens."""
    text = html.unescape(text)               # HTML entities are decoded so "&amp;" becomes "&".
    text = text.translate(_QUOTES)           # Curly quotes are replaced by straight quotes so that both spellings match.
    if config.squeeze_elongation:
        text = _ELONGATION_RE.sub(r"\1\1", text)
    if config.lowercase:
        text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    if config.hashtags == "strip":
        tokens = [token[1:] if token.startswith("#") else token for token in tokens]
    if not config.keep_punct:
        tokens = [token for token in tokens if token not in {"!", "?", "…"}]
    if not config.keep_emoji:
        tokens = [token for token in tokens if not EMOJI_RE.fullmatch(token)]
    return tokens


def prepare_text(text, config=DEFAULT_CONFIG):
    """Clean one tweet and return its tokens joined by spaces (never an empty string)."""
    tokens = tokenize(text, config)
    return " ".join(tokens) if tokens else "<empty>"


def prepare_series(texts, config=DEFAULT_CONFIG):
    """Apply prepare_text() to every tweet in a pandas Series."""
    return texts.map(lambda text: prepare_text(text, config))
