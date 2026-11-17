from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import nsmallest
from math import isfinite, log10, log1p, sqrt

from .indice import IndiceInvertido
from .preprocesamiento import Preprocesador

MODELOS = ("binario", "tfidf", "bm25")


@dataclass(frozen=True, slots=True)
class Resultado:
    doc_id: str
    puntaje: float


class BuscadorClasico:
    def __init__(
        self, indice: IndiceInvertido, preprocesador: Preprocesador,
        *, k1: float = 1.2, b: float = 0.75,
    ):
        if not isfinite(k1) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requiere k1 finito > 0 y 0 <= b <= 1.")

        self.indice = indice
        self.preprocesador = preprocesador
        n = len(indice.longitudes)
        self._idf_tfidf = {}
        self._idf_bm25 = {}
        norma_cuadrada = defaultdict(float)

        for termino, posting in indice.postings.items():
            df = len(posting)
            idf = log10(n / df)
            self._idf_tfidf[termino] = idf
            self._idf_bm25[termino] = log1p((n - df + 0.5) / (df + 0.5))
            for doc_id, tf in posting.items():
                norma_cuadrada[doc_id] += (tf * idf) ** 2

        self._norma_doc = {doc_id: sqrt(valor) for doc_id, valor in norma_cuadrada.items()}
        self._k1_mas_uno = k1 + 1
        self._ajuste_bm25 = {}
        if indice.longitud_media > 0:
            self._ajuste_bm25 = {
                doc_id: k1 * (1 - b + b * largo / indice.longitud_media)
                for doc_id, largo in indice.longitudes.items()
            }

    def buscar(self, consulta: str, modelo: str = "bm25", top_n: int = 10) -> list[Resultado]:
        """Recupera hasta top_n resultados positivos, ordenados por puntaje e ID."""
        if modelo not in MODELOS:
            raise ValueError(f"Modelo inválido: {modelo}. Usa uno de {MODELOS}.")
        if top_n <= 0:
            raise ValueError("top_n debe ser positivo.")

        consulta_tf = Counter(self.preprocesador.procesar(consulta))
        if modelo == "binario":
            puntajes = self._jaccard(consulta_tf)
        elif modelo == "tfidf":
            puntajes = self._coseno(consulta_tf)
        else:
            puntajes = self._bm25(consulta_tf)

        mejores = nsmallest(top_n, puntajes.items(), key=lambda par: (-par[1], par[0]))
        return [Resultado(doc_id, puntaje) for doc_id, puntaje in mejores]

    def _jaccard(self, consulta_tf: Counter[str]) -> dict[str, float]:
        # Proyecta la consulta sobre el vocabulario del índice binario.
        postings = self.indice.postings
        intersecciones = Counter()
        cantidad = 0
        for termino in consulta_tf:
            posting = postings.get(termino)
            if posting is not None:
                cantidad += 1
                intersecciones.update(posting.keys())

        return {
            doc_id: comunes / (cantidad + self.indice.terminos_distintos[doc_id] - comunes)
            for doc_id, comunes in intersecciones.items()
        }

    def _coseno(self, consulta_tf: Counter[str]) -> dict[str, float]:
        productos = defaultdict(float)
        norma_consulta_cuadrada = 0.0
        for termino, tf_consulta in consulta_tf.items():
            idf = self._idf_tfidf.get(termino, 0.0)
            if idf == 0:
                continue
            peso_consulta = tf_consulta * idf
            norma_consulta_cuadrada += peso_consulta ** 2
            for doc_id, tf_doc in self.indice.postings[termino].items():
                productos[doc_id] += peso_consulta * (tf_doc * idf)

        norma_consulta = sqrt(norma_consulta_cuadrada)
        return {
            doc_id: producto / (norma_consulta * self._norma_doc[doc_id])
            for doc_id, producto in productos.items()
        }

    def _bm25(self, consulta_tf: Counter[str]) -> dict[str, float]:
        puntajes = defaultdict(float)
        for termino in consulta_tf:
            posting = self.indice.postings.get(termino)
            if posting is None:
                continue
            idf = self._idf_bm25[termino]
            for doc_id, tf in posting.items():
                puntajes[doc_id] += idf * (tf * self._k1_mas_uno) / (tf + self._ajuste_bm25[doc_id])
        return dict(puntajes)
