import argparse
from pathlib import Path

from fiqa_ri.catalogo import Catalogo, preparar_catalogo
from fiqa_ri.datos import cargar_fiqa
from fiqa_ri.embeddings import BuscadorSemantico, preparar_faiss, preparar_modelo
from fiqa_ri.evaluacion import completar_evaluacion, evaluar_clasicos
from fiqa_ri.modelos import BuscadorClasico
from fiqa_ri.preprocesamiento import Preprocesador


def main():
    parser = argparse.ArgumentParser(description="Prepara los recursos de la aplicación FiQA.")
    parser.add_argument(
        "etapa", choices=("todo", "catalogo", "modelo", "embeddings", "evaluacion"),
        nargs="?", default="todo",
    )
    parser.add_argument("--assets", type=Path, default=Path(__file__).parent / "assets")
    parser.add_argument(
        "--datos",
        type=Path,
        default=None,
        help="Ruta a la carpeta FIQA que contiene corpus.jsonl, queries.jsonl y qrels/",
    )
    args = parser.parse_args()
    if args.etapa in ("todo", "catalogo"):
        preparar_catalogo(args.assets / "catalogo.sqlite", args.datos)
    if args.etapa in ("todo", "modelo", "embeddings"):
        preparar_modelo(args.assets / "modelo")
    if args.etapa in ("todo", "embeddings"):
        preparar_faiss(cargar_fiqa("test", args.datos).documentos.values(), args.assets)
    if args.etapa in ("todo", "evaluacion"):
        catalogo = Catalogo(args.assets / "catalogo.sqlite")
        clasico = BuscadorClasico(catalogo.indice(), Preprocesador())
        metricas_clasicas = evaluar_clasicos(catalogo, clasico)
        completar_evaluacion(args.assets, catalogo, metricas_clasicas, BuscadorSemantico(args.assets))


if __name__ == "__main__":
    main()
