from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .datos import Documento
from .preprocesamiento import Preprocesador


@dataclass(slots=True)
class IndiceInvertido:
    postings: Mapping[str, dict[str, int]]
    longitudes: dict[str, int]
    terminos_distintos: dict[str, int]
    longitud_media: float


def construir_indice(
    documentos: Iterable[Documento], preprocesador: Preprocesador
) -> IndiceInvertido:
    documentos = list(documentos)
    textos = (f"{doc.titulo}\n{doc.texto}" for doc in documentos)
    postings = defaultdict(dict)
    longitudes = {}
    terminos_distintos = {}

    for doc, tokens in zip(documentos, preprocesador.procesar_lote(textos), strict=True):
        if doc.id in longitudes:
            raise ValueError(f"Documento duplicado: {doc.id}")
        frecuencias = Counter(tokens)
        longitudes[doc.id] = len(tokens)
        terminos_distintos[doc.id] = len(frecuencias)
        for termino, frecuencia in frecuencias.items():
            postings[termino][doc.id] = frecuencia

    longitud_media = sum(longitudes.values()) / len(longitudes) if longitudes else 0.0
    return IndiceInvertido(dict(postings), longitudes, terminos_distintos, longitud_media)
