"""Índice de pasajes FiQA con MiniLM y FAISS."""
import json
from pathlib import Path

MODEL_ID = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
MODEL_REVISION = "14ca9be4bbcf1402eac0f43a2e2ccb6e0f994ba3"
MODEL_FILE = "onnx/model_quint8_avx2.onnx"
MAX_TOKENS = 128


def crear_codificador():
    import onnxruntime as ort
    from sentence_transformers import SentenceTransformer

    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = 4
    encoder = SentenceTransformer(
        MODEL_ID, revision=MODEL_REVISION, device="cpu", backend="onnx",
        model_kwargs={
            "file_name": MODEL_FILE,
            "provider": "CPUExecutionProvider",
            "session_options": opciones,
        },
    )
    encoder.max_seq_length = MAX_TOKENS
    return encoder


def preparar_faiss(documentos, destino: Path) -> None:
    import faiss

    # Agrupar textos de longitudes semejantes reduce el relleno de cada lote.
    filas = sorted(documentos, key=lambda doc: (len(doc.titulo) + len(doc.texto), doc.id))
    textos = [f"{doc.titulo}\n{doc.texto}".strip() for doc in filas]
    encoder = crear_codificador()
    indice = faiss.IndexScalarQuantizer(
        encoder.get_embedding_dimension(), faiss.ScalarQuantizer.QT_fp16,
        faiss.METRIC_INNER_PRODUCT,
    )
    for inicio in range(0, len(textos), 1024):
        vectores = encoder.encode(
            textos[inicio:inicio + 1024], batch_size=32, normalize_embeddings=True,
            show_progress_bar=False,
        )
        indice.add(vectores)
        if inicio % 4096 == 0:
            print(f"Embeddings: {indice.ntotal:,}/{len(textos):,}", flush=True)

    destino.mkdir(parents=True, exist_ok=True)
    faiss.write_index(indice, str(destino / "pasajes.faiss"))
    (destino / "ids.json").write_text(json.dumps([doc.id for doc in filas]), encoding="utf-8")


def preparar_modelo(destino: Path) -> None:
    import shutil
    from urllib.request import urlopen

    destino.mkdir(parents=True, exist_ok=True)
    for archivo in ("tokenizer.json", MODEL_FILE):
        salida = destino / Path(archivo).name
        temporal = salida.with_suffix(".tmp")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{archivo}"
        with urlopen(url, timeout=60) as respuesta, temporal.open("wb") as stream:
            shutil.copyfileobj(respuesta, stream)
        temporal.replace(salida)


class BuscadorSemantico:
    def __init__(self, assets: Path):
        import faiss
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.ids = json.loads((assets / "ids.json").read_text(encoding="utf-8"))
        self.indice = faiss.read_index(str(assets / "pasajes.faiss"))
        if self.indice.ntotal != len(self.ids):
            raise ValueError("El índice FAISS y sus IDs no corresponden.")

        self.tokenizer = Tokenizer.from_file(str(assets / "modelo/tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        config = ort.SessionOptions()
        config.intra_op_num_threads = 1
        self.sesion = ort.InferenceSession(
            str(assets / "modelo" / Path(MODEL_FILE).name),
            sess_options=config, providers=["CPUExecutionProvider"],
        )
        self.entradas = {entrada.name for entrada in self.sesion.get_inputs()}

    def codificar(self, textos):
        import numpy as np

        tokens = self.tokenizer.encode_batch(textos)
        mascara = np.array([token.attention_mask for token in tokens], dtype=np.int64)
        entradas = {
            "input_ids": np.array([token.ids for token in tokens], dtype=np.int64),
            "attention_mask": mascara,
            "token_type_ids": np.array([token.type_ids for token in tokens], dtype=np.int64),
        }
        vectores = self.sesion.run(None, {nombre: entradas[nombre] for nombre in self.entradas})[0]
        pesos = mascara[..., None].astype(np.float32)
        medios = (vectores * pesos).sum(axis=1) / pesos.sum(axis=1).clip(min=1)
        return medios / np.linalg.norm(medios, axis=1, keepdims=True).clip(min=1e-12)

    def buscar_lote(self, consultas: list[str], top_n: int):
        from .modelos import Resultado

        if not consultas:
            return []
        puntajes, posiciones = self.ranking(consultas, top_n)
        return [
            [Resultado(self.ids[pos], float(score)) for pos, score in zip(indices, scores) if pos >= 0]
            for indices, scores in zip(posiciones, puntajes)
        ]

    def buscar(self, consulta: str, top_n: int):
        return self.buscar_lote([consulta], top_n)[0] if consulta.strip() else []

    def ranking(self, consultas: list[str], top_n: int):
        import numpy as np

        # ONNX cuantiza dinámicamente las activaciones. Codificar por separado
        # mantiene la misma entrada al comparar consultas web y de evaluación.
        vectores = np.concatenate([self.codificar([consulta]) for consulta in consultas])
        return self.indice.search(vectores, top_n)
