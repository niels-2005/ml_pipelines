import re
import unicodedata

import demoji
import nltk
import pandas as pd
from contractions import fix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# Erforderliche Downloads für NLTK
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class TextCleaner:
    def __init__(self, df: pd.DataFrame, text_col: str):
        self.df = df
        self.col = text_col
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def _apply_to_column(self, func, *args, **kwargs):
        """Applies a function to the specified text column."""
        self.df[self.col] = self.df[self.col].apply(
            lambda text: func(text, *args, **kwargs)
        )

    def lower_text(self):
        """Converts text to lowercase."""
        self._apply_to_column(str.lower)

    def remove_stopwords(self):
        """Removes stopwords from the text."""
        self._apply_to_column(
            lambda text: " ".join(
                [w for w in word_tokenize(text) if w.lower() not in self.stop_words]
            )
        )

    def remove_emojis(self):
        """Removes emojis from the text."""
        self._apply_to_column(demoji.replace)

    def perform_lemmatization(self):
        """Performs lemmatization on the text."""
        self._apply_to_column(
            lambda text: " ".join(
                [self.lemmatizer.lemmatize(w) for w in word_tokenize(text)]
            )
        )

    def remove_numbers(self):
        """Removes numbers from the text."""
        self._apply_to_column(lambda text: re.sub(r"\d+", "", text))

    def remove_punctuation(self):
        """Removes punctuation from the text."""
        self._apply_to_column(lambda text: re.sub(r"[^\w\s]", "", text))

    def remove_urls(self):
        """Removes URLs from the text."""
        self._apply_to_column(lambda text: re.sub(r"http\S+|www\S+", "", text))

    def remove_emails(self):
        """Removes email addresses from the text."""
        self._apply_to_column(lambda text: re.sub(r"\S+@\S+", "", text))

    def remove_html_tags(self):
        """Removes HTML tags from the text."""
        self._apply_to_column(lambda text: re.sub(r"<.*?>", "", text))

    def remove_extra_spaces(self):
        """Removes extra spaces from the text."""
        self._apply_to_column(lambda text: re.sub(r"\s+", " ", text).strip())

    def remove_currency_symbols(self):
        """Removes currency symbols and special characters like %, &, #, etc."""
        self._apply_to_column(lambda text: re.sub(r"[$€£%&@#]+", "", text))

    def reduce_repeated_chars(self):
        """Reduces excessive repeated characters (e.g., loooove -> loove)."""
        self._apply_to_column(lambda text: re.sub(r"(.)\1{2,}", r"\1\1", text))

    def remove_non_english_chars(self):
        """Removes non-English characters from the text."""
        self._apply_to_column(lambda text: re.sub(r"[^\x00-\x7F]+", "", text))

    def remove_hashtags_mentions(self):
        """Removes hashtags and mentions (e.g., @user, #topic)."""
        self._apply_to_column(lambda text: re.sub(r"#\S+|@\S+", "", text))

    def remove_accented_chars(self):
        """Removes accented characters (e.g., é -> e, ü -> u)."""
        self._apply_to_column(
            lambda text: unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )

    def remove_short_words(self, min_length: int = 3):
        """Removes words shorter than `min_length` characters."""
        self._apply_to_column(
            lambda text: " ".join(
                [word for word in text.split() if len(word) >= min_length]
            )
        )

    def correct_spelling(self):
        """Corrects spelling errors in the text using TextBlob."""
        self._apply_to_column(lambda text: str(TextBlob(text).correct()))

    def expand_contractions(self):
        """Expands contractions (e.g., don't -> do not)."""
        self._apply_to_column(fix)
