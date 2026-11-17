from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Documento:
    id: str
    titulo: str
    texto: str


@dataclass(slots=True)
class FiQA:
    documentos: dict[str, Documento]
    consultas: dict[str, str]
    qrels: dict[str, dict[str, int]]


SPLITS = ("train", "validation", "test")
QREL_FILES = {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"}


def resolver_ruta_fiqa(ruta: str | Path | None = None) -> Path:
    """Encuentra el corpus FIQA local sin depender de Hugging Face."""
    candidatos = []
    if ruta is not None:
        candidatos.append(Path(ruta))
    if os.environ.get("FIQA_DATA_DIR"):
        candidatos.append(Path(os.environ["FIQA_DATA_DIR"]))

    base = Path(__file__).resolve().parents[1]
    candidatos.extend((
        base / "datos" / "fiqa",
        Path.cwd() / "datos" / "fiqa",
        Path.cwd() / "fiqa",
        Path.home() / "Downloads" / "fiqa" / "fiqa",
        Path.home() / "Downloads" / "fiqa (1)" / "fiqa",
    ))

    for candidato in candidatos:
        candidato = candidato.expanduser().resolve()
        if (candidato / "corpus.jsonl").is_file() and (candidato / "queries.jsonl").is_file():
            return candidato
    raise FileNotFoundError(
        "No se encontró el corpus FIQA local. Define FIQA_DATA_DIR con la carpeta "
        "que contiene corpus.jsonl, queries.jsonl y qrels/, o coloca los datos en "
        f"{base / 'datos' / 'fiqa'}."
    )


def cargar_fiqa(split: str = "test", ruta: str | Path | None = None) -> FiQA:
    """Carga el corpus, las consultas y los juicios desde los archivos locales."""
    if split not in SPLITS:
        raise ValueError(f"Split inválido: {split}. Usa uno de {SPLITS}.")

    directorio = resolver_ruta_fiqa(ruta)
    documentos = {}
    with (directorio / "corpus.jsonl").open("r", encoding="utf-8") as archivo:
        for linea in archivo:
            fila = json.loads(linea)
            doc_id = str(fila["_id"])
            documentos[doc_id] = Documento(doc_id, fila.get("title") or "", fila.get("text") or "")

    textos_consultas = {}
    with (directorio / "queries.jsonl").open("r", encoding="utf-8") as archivo:
        for linea in archivo:
            fila = json.loads(linea)
            textos_consultas[str(fila["_id"])] = fila["text"]

    consultas = {}
    qrels = {}
    archivo_qrels = directorio / "qrels" / QREL_FILES[split]
    if not archivo_qrels.is_file():
        raise FileNotFoundError(f"Falta el archivo de juicios FIQA: {archivo_qrels}")

    with archivo_qrels.open("r", encoding="utf-8") as archivo:
        primera = archivo.readline().strip()
        if primera.lower().startswith("query-id"):
            primera = ""
        for linea in ([primera] if primera else []) + archivo.readlines():
            if not linea.strip():
                continue
            partes = linea.strip().split("\t")
            if len(partes) >= 3:
                qid, doc_id, score = partes[0], partes[1], int(partes[2])
                if qid not in textos_consultas or doc_id not in documentos:
                    raise ValueError(
                        f"Qrel sin consulta o documento en FIQA: qid={qid}, doc_id={doc_id}"
                    )
                consultas[qid] = textos_consultas[qid]
                qrels.setdefault(qid, {})[doc_id] = score

    return FiQA(documentos, consultas, qrels)
