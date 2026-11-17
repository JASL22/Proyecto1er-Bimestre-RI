import argparse
from time import perf_counter

from .datos import SPLITS, cargar_fiqa
from .indice import construir_indice
from .modelos import MODELOS, BuscadorClasico
from .preprocesamiento import MODOS, Preprocesador


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa FIQA y consulta los modelos clásicos.")
    origen = parser.add_mutually_exclusive_group()
    origen.add_argument("--consulta", help="Consulta libre en inglés.")
    origen.add_argument("--qid", help="ID de una consulta oficial del split elegido.")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--modo", choices=MODOS, default="stemming")
    parser.add_argument("--modelo", choices=(*MODELOS, "todos"), default="todos")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--conservar-stopwords", action="store_true")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top debe ser positivo.")

    preprocesador = Preprocesador(args.modo, quitar_stopwords=not args.conservar_stopwords)
    print(f"Cargando FIQA y los qrels de {args.split}…", flush=True)
    fiqa = cargar_fiqa(args.split)
    if args.qid is not None and args.qid not in fiqa.consultas:
        parser.error(f"El ID {args.qid!r} no tiene juicios en {args.split}.")

    inicio = perf_counter()
    print(f"Construyendo el índice con {args.modo}…", flush=True)
    indice = construir_indice(fiqa.documentos.values(), preprocesador)
    buscador = BuscadorClasico(indice, preprocesador, k1=args.k1, b=args.b)
    print(
        f"{len(fiqa.documentos):,} documentos · {len(indice.postings):,} términos · "
        f"{len(fiqa.consultas):,} consultas juzgadas · índice listo en {perf_counter() - inicio:.1f} s."
    )

    qid = args.qid
    consulta = args.consulta
    if consulta is None:
        if qid is None:
            qid = next(iter(fiqa.consultas))
        consulta = fiqa.consultas[qid]
    etiqueta = f"Consulta {qid}" if qid is not None else "Consulta libre"
    print(f"\n{etiqueta}: {consulta}")

    modelos = MODELOS if args.modelo == "todos" else (args.modelo,)
    for modelo in modelos:
        print(f"\n{modelo.upper()}")
        resultados = buscador.buscar(consulta, modelo, args.top)
        if not resultados:
            print("Sin coincidencias con puntaje positivo.")
        for puesto, resultado in enumerate(resultados, 1):
            doc = fiqa.documentos[resultado.doc_id]
            extracto = " ".join(f"{doc.titulo} {doc.texto}".split())
            if len(extracto) > 150:
                extracto = extracto[:149] + "…"
            print(f"{puesto:2}. [{doc.id}] {resultado.puntaje:.6f}  {extracto}")


if __name__ == "__main__":
    main()
