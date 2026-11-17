"""Persistencia local del índice invertido y del catálogo de FiQA."""
import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from zlib import compress, decompress

from .datos import Documento
from .indice import IndiceInvertido

SPLITS_EVALUACION = ("test", "validation")


class Catalogo:
    def __init__(self, archivo: Path):
        self.uri = archivo.resolve().as_uri() + "?mode=ro"

    def conectar(self):
        return sqlite3.connect(self.uri, uri=True)

    def indice(self) -> IndiceInvertido:
        with closing(self.conectar()) as db:
            estadisticas = list(db.execute("SELECT id, largo, unicos FROM documentos"))
            cantidad = db.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        longitudes = {doc_id: largo for doc_id, largo, _ in estadisticas}
        unicos = {doc_id: distintos for doc_id, _, distintos in estadisticas}
        media = sum(longitudes.values()) / len(longitudes) if longitudes else 0.0
        return IndiceInvertido(PostingsSQLite(self, cantidad), longitudes, unicos, media)

    def documentos(self, ids: set[str]) -> dict[str, Documento]:
        if not ids:
            return {}
        marcadores = ",".join("?" for _ in ids)
        with closing(self.conectar()) as db:
            cursor = db.execute(
                f"SELECT id, titulo, texto FROM documentos WHERE id IN ({marcadores})", tuple(ids)
            )
            return {row[0]: Documento(*row) for row in cursor}

    def consultas(self, split: str) -> dict[str, str]:
        with closing(self.conectar()) as db:
            return dict(db.execute(
                "SELECT id, texto FROM consultas WHERE split = ? ORDER BY id", (split,)
            ))

    def juicios(self, split: str, qid: str) -> dict[str, int]:
        with closing(self.conectar()) as db:
            return dict(db.execute(
                "SELECT doc_id, score FROM qrels WHERE split = ? AND qid = ?", (split, qid)
            ))

    def qrels(self, split: str) -> dict[str, dict[str, int]]:
        datos = {}
        with closing(self.conectar()) as db:
            for qid, doc_id, score in db.execute(
                "SELECT qid, doc_id, score FROM qrels WHERE split = ?", (split,)
            ):
                datos.setdefault(qid, {})[doc_id] = score
        return datos


class PostingsSQLite(Mapping):
    def __init__(self, catalogo: Catalogo, cantidad: int):
        self.catalogo = catalogo
        self.cantidad = cantidad

    def __len__(self):
        return self.cantidad

    def __iter__(self):
        with closing(self.catalogo.conectar()) as db:
            for termino, in db.execute("SELECT termino FROM postings"):
                yield termino

    def __getitem__(self, termino: str):
        with closing(self.catalogo.conectar()) as db:
            fila = db.execute("SELECT datos FROM postings WHERE termino = ?", (termino,)).fetchone()
        if fila is None:
            raise KeyError(termino)
        return json.loads(decompress(fila[0]))

    def items(self):
        with closing(self.catalogo.conectar()) as db:
            for termino, datos in db.execute("SELECT termino, datos FROM postings ORDER BY orden"):
                yield termino, json.loads(decompress(datos))


def preparar_catalogo(archivo: Path, ruta_fiqa=None) -> None:
    from .datos import cargar_fiqa
    from .indice import construir_indice
    from .preprocesamiento import Preprocesador

    archivo.parent.mkdir(parents=True, exist_ok=True)
    temporal = archivo.with_suffix(".tmp")
    temporal.unlink(missing_ok=True)
    fiqa = cargar_fiqa("test", ruta_fiqa)
    indice = construir_indice(fiqa.documentos.values(), Preprocesador())

    with closing(sqlite3.connect(temporal)) as db, db:
        db.executescript("""
            CREATE TABLE documentos(
                id TEXT PRIMARY KEY, titulo TEXT, texto TEXT, largo INTEGER, unicos INTEGER
            );
            CREATE TABLE postings(orden INTEGER PRIMARY KEY, termino TEXT UNIQUE, datos BLOB);
            CREATE TABLE consultas(split TEXT, id TEXT, texto TEXT, PRIMARY KEY(split, id));
            CREATE TABLE qrels(
                split TEXT, qid TEXT, doc_id TEXT, score INTEGER, PRIMARY KEY(split, qid, doc_id)
            );
        """)
        db.executemany("INSERT INTO documentos VALUES (?, ?, ?, ?, ?)", (
            (doc.id, doc.titulo, doc.texto, indice.longitudes[doc.id], indice.terminos_distintos[doc.id])
            for doc in fiqa.documentos.values()
        ))
        db.executemany("INSERT INTO postings(termino, datos) VALUES (?, ?)", (
            (termino, compress(json.dumps(posting, separators=(",", ":")).encode(), level=3))
            for termino, posting in indice.postings.items()
        ))
        for split in SPLITS_EVALUACION:
            juzgados = fiqa if split == "test" else cargar_fiqa(split, ruta_fiqa)
            db.executemany("INSERT INTO consultas VALUES (?, ?, ?)", (
                (split, qid, texto) for qid, texto in juzgados.consultas.items()
            ))
            db.executemany("INSERT INTO qrels VALUES (?, ?, ?, ?)", (
                (split, qid, doc_id, score)
                for qid, juicios in juzgados.qrels.items() for doc_id, score in juicios.items()
            ))
    temporal.replace(archivo)
