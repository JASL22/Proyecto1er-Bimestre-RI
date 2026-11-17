import re
import unicodedata
from collections.abc import Iterable, Iterator
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

MODOS = ("stemming", "lema")
TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)?")
NEGACIONES = {"no", "not", "nor", "never"}


class Preprocesador:
    """Aplica la misma configuración a documentos y consultas."""

    def __init__(self, modo: str = "stemming", quitar_stopwords: bool = True):
        if modo not in MODOS:
            raise ValueError(f"Modo inválido: {modo}. Usa uno de {MODOS}.")

        try:
            self._stopwords = set(stopwords.words("english")) - NEGACIONES if quitar_stopwords else set()
        except LookupError as error:
            raise RuntimeError("Ejecuta python -m nltk.downloader stopwords.") from error

        self._nlp = None
        if modo == "lema":
            try:
                import spacy
            except ImportError as error:
                raise RuntimeError('Instala spaCy con python -m pip install "spacy>=3.8,<4".') from error
            try:
                self._nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
            except OSError as error:
                raise RuntimeError("Ejecuta python -m spacy download en_core_web_sm.") from error
        else:
            self._raiz = lru_cache(maxsize=65_536)(SnowballStemmer("english").stem)

    def procesar_lote(self, textos: Iterable[str]) -> Iterator[list[str]]:
        textos = (unicodedata.normalize("NFKC", texto).replace("’", "'") for texto in textos)
        if self._nlp is None:
            for texto in textos:
                tokens = TOKEN.findall(texto.lower())
                yield [
                    self._raiz(token) if token.isalpha() or "'" in token else token
                    for token in tokens
                    if token not in self._stopwords
                ]
            return

        # El modelo recibe el texto con mayúsculas y contexto antes de normalizarlo.
        for doc in self._nlp.pipe(textos, batch_size=64):
            terminos = []
            for token in doc:
                if token.is_space or token.is_punct:
                    continue
                forma = token.lemma_ or token.text
                terminos.extend(
                    termino for termino in TOKEN.findall(forma.lower())
                    if termino not in self._stopwords
                )
            yield terminos

    def procesar(self, texto: str) -> list[str]:
        return next(self.procesar_lote((texto,)))
