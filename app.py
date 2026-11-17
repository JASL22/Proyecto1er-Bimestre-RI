import json
import os
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, render_template, request

from fiqa_ri.catalogo import SPLITS_EVALUACION, Catalogo
from fiqa_ri.embeddings import BuscadorSemantico
from fiqa_ri.evaluacion import MODELOS_WEB, metricas, resumen
from fiqa_ri.modelos import BuscadorClasico
from fiqa_ri.preprocesamiento import Preprocesador

ETIQUETAS = {
    "binario": "Binario · Jaccard",
    "tfidf": "Vectorial · TF-IDF",
    "bm25": "BM25",
    "embeddings": "Embeddings",
}
TOP_N = 10


def create_app(assets: Path | None = None) -> Flask:
    app = Flask(__name__)
    assets = assets or Path(os.environ.get("FIQA_ASSETS", Path(__file__).parent / "assets"))
    catalogo = Catalogo(assets / "catalogo.sqlite")
    app.jinja_env.globals.update(etiquetas=ETIQUETAS, modelos=MODELOS_WEB)

    @lru_cache(maxsize=1)
    def clasico():
        return BuscadorClasico(catalogo.indice(), Preprocesador())

    @lru_cache(maxsize=1)
    def semantico():
        return BuscadorSemantico(assets)

    @lru_cache(maxsize=1)
    def datos_evaluacion():
        ruta = assets / "evaluacion.json"
        return json.loads(ruta.read_text(encoding="utf-8"))

    @lru_cache(maxsize=16)
    def resumen_split(split, k):
        datos = datos_evaluacion()[split]
        return {modelo: resumen(datos, modelo, k) for modelo in MODELOS_WEB}

    def split_oficial():
        split = request.args.get("split", "test")
        if split not in SPLITS_EVALUACION:
            abort(400, "Elige uno de los conjuntos oficiales disponibles.")
        return split

    @app.get("/health")
    def salud():
        archivos = ("catalogo.sqlite", "pasajes.faiss", "ids.json", "evaluacion.json",
                    "modelo/model_quint8_avx2.onnx", "modelo/tokenizer.json")
        if not all((assets / archivo).is_file() for archivo in archivos):
            return {"estado": "preparando recursos"}, 503
        try:
            motor_clasico = clasico()
            motor_semantico = semantico()
            datos_evaluacion()
            if len(motor_clasico.indice.longitudes) != len(motor_semantico.ids):
                raise ValueError("Los índices deben cubrir el mismo número de documentos.")
        except Exception:
            app.logger.exception("Error al cargar los recursos de FiQA.")
            return {"estado": "error en los recursos"}, 503
        return {"estado": "listo"}

    @app.get("/")
    def buscar():
        consulta = request.args.get("q", "").strip()
        modelo = request.args.get("modelo", "bm25")
        comparar = request.args.get("comparar") == "1"
        if modelo not in ETIQUETAS or len(consulta) > 500:
            abort(400, "El modelo debe ser válido y la consulta puede tener hasta 500 caracteres.")
        k = TOP_N
        split = split_oficial()
        qid = request.args.get("qid")
        juicios = {}
        if qid:
            consulta_oficial = catalogo.consultas(split).get(qid)
            if consulta_oficial is None:
                abort(404, "Consulta oficial no disponible.")
            if not consulta or consulta == consulta_oficial.strip():
                consulta = consulta_oficial
                juicios = catalogo.juicios(split, qid)
            else:
                qid = None

        resultados = {}
        documentos = {}
        if consulta:
            seleccion = MODELOS_WEB if comparar else (modelo,)
            for nombre in seleccion:
                resultados[nombre] = (
                    semantico().buscar(consulta, k) if nombre == "embeddings"
                    else clasico().buscar(consulta, nombre, k)
                )
            documentos = catalogo.documentos({r.doc_id for fila in resultados.values() for r in fila})
        return render_template(
            "buscar.html", pagina="buscar", consulta=consulta, modelo=modelo, k=k, qid=qid,
            comparar=comparar,
            split=split, juicios=juicios, resultados=resultados, documentos=documentos,
        )

    @app.get("/evaluacion")
    def evaluacion():
        split, k = split_oficial(), TOP_N
        consultas = catalogo.consultas(split)
        qid = request.args.get("qid", next(iter(consultas)))
        if qid not in consultas:
            abort(404, "Consulta oficial no disponible.")
        preparados = (assets / "evaluacion.json").is_file()
        globales = resumen_split(split, k) if preparados else None
        registro = datos_evaluacion()[split][qid] if preparados else None
        por_consulta = {
            nombre: metricas(ranks, registro["total"], k)
            for nombre, ranks in registro["modelos"].items()
        } if registro else None
        return render_template(
            "evaluacion.html", pagina="evaluacion", split=split, k=k, consultas=consultas,
            qid=qid, globales=globales, por_consulta=por_consulta,
            total_relevantes=registro["total"] if registro else None,
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
