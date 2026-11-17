"""Precision, Recall y Average Precision con qrels oficiales."""
import json
from itertools import batched
from pathlib import Path
from statistics import fmean

from .catalogo import SPLITS_EVALUACION, Catalogo
from .modelos import MODELOS, BuscadorClasico

MODELOS_WEB = (*MODELOS, "embeddings")


def metricas(posiciones: list[int], relevantes: int, k: int) -> dict[str, float]:
    aciertos_k = sum(puesto <= k for puesto in posiciones)
    return {
        "precision": aciertos_k / k,
        "recall": aciertos_k / relevantes if relevantes else 0.0,
        "ap": sum(i / puesto for i, puesto in enumerate(posiciones, 1)) / relevantes
        if relevantes else 0.0,
    }


def resumen(consultas: dict, modelo: str, k: int) -> dict[str, float]:
    valores = [
        metricas(registro["modelos"][modelo], registro["total"], k)
        for registro in consultas.values()
    ]
    return {clave: fmean(fila[clave] for fila in valores) for clave in valores[0]}


def evaluar_clasicos(catalogo: Catalogo, clasico: BuscadorClasico) -> dict:
    total_docs = len(clasico.indice.longitudes)
    evaluacion = {}
    for split in SPLITS_EVALUACION:
        consultas = catalogo.consultas(split)
        qrels = catalogo.qrels(split)
        juzgados = {qid: {doc_id for doc_id, score in qrels[qid].items() if score > 0} for qid in consultas}
        filas = {qid: {"total": len(relevantes), "modelos": {}} for qid, relevantes in juzgados.items()}
        for modelo in MODELOS:
            for qid, texto in consultas.items():
                ranking = clasico.buscar(texto, modelo, total_docs)
                filas[qid]["modelos"][modelo] = [
                    i for i, res in enumerate(ranking, 1) if res.doc_id in juzgados[qid]
                ]
            print(f"{split}: {modelo} evaluado sobre {len(consultas)} consultas.", flush=True)
        evaluacion[split] = filas
    return evaluacion


def completar_evaluacion(assets: Path, catalogo: Catalogo, evaluacion: dict, semantico) -> None:
    import numpy as np

    total_docs = len(semantico.ids)
    posiciones_faiss = {doc_id: i for i, doc_id in enumerate(semantico.ids)}
    if set(posiciones_faiss) != catalogo.indice().longitudes.keys():
        raise ValueError("FAISS y el índice léxico deben cubrir el mismo corpus.")

    for split, filas in evaluacion.items():
        consultas = catalogo.consultas(split)
        qrels = catalogo.qrels(split)
        for lote in batched(consultas, 8):
            _, etiquetas = semantico.ranking([consultas[qid] for qid in lote], total_docs)
            for qid, ranking in zip(lote, etiquetas, strict=True):
                esperados = [posiciones_faiss[doc_id] for doc_id, score in qrels[qid].items() if score > 0]
                aciertos = np.flatnonzero(np.isin(ranking, esperados)) + 1
                filas[qid]["modelos"]["embeddings"] = aciertos.tolist()
        print(f"{split}: embeddings evaluado sobre {len(consultas)} consultas.", flush=True)

    salida = assets / "evaluacion.json"
    temporal = salida.with_suffix(".tmp")
    temporal.write_text(json.dumps(evaluacion, separators=(",", ":")), encoding="utf-8")
    temporal.replace(salida)
