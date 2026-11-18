# Recuperación de Información · Proyecto de primer bimestre

## FiQA · Buscador financiero

Aplicación Flask para buscar sobre los 57.638 documentos de BEIR FiQA en inglés. Permite consultar uno de los cuatro modelos, comparar sus rankings y evaluar consultas oficiales en validación y prueba.

El proyecto cumple el objetivo de comparar recuperación clásica basada en términos y recuperación semántica con embeddings. La interfaz mantiene una búsqueda sencilla y ofrece selección de modelo, comparación de rankings y métricas de evaluación.

## Ejecutar en Windows

Descomprime el proyecto y ejecuta desde la carpeta que contiene `app.py`, con Python 3.12 o 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m nltk.downloader stopwords
.\.venv\Scripts\python.exe preparar.py modelo
.\.venv\Scripts\python.exe app.py
```

El proyecto carga los archivos locales FIQA (`corpus.jsonl`, `queries.jsonl` y `qrels/`) y no necesita descargar el corpus al ejecutarse.
Busca automáticamente `Downloads\fiqa\fiqa` y `Downloads\fiqa (1)\fiqa`.
Puedes fijar otra copia con `FIQA_DATA_DIR` o con `preparar.py ... --datos RUTA`.

Abre [el buscador local](http://127.0.0.1:8000/). En Linux o macOS usa `.venv/bin/python` en lugar del ejecutable de Windows. La búsqueda mantiene solo el campo de texto, el selector de uno de los cuatro modelos y el botón Buscar. Los resultados se muestran en Top 10; desde ellos se puede abrir la comparación de los cuatro modelos. La sección Evaluación muestra precisión y exhaustividad en los primeros 10 resultados, AP por consulta y MAP global. El modelo ONNX incluido está dirigido a equipos x86-64 con AVX2.

## Publicación en Render Free

El repositorio Git debe incluir `assets/`, además del código. Crea un **Blueprint** de Render usando ese repositorio y confirma el servicio `fiqa-ri-epn` con plan **Free**. Render toma los comandos de `render.yaml`. La comprobación `/health` responde con código 200 cuando la aplicación puede cargar los índices, el codificador y las métricas. [Guía de Blueprints de Render](https://render.com/docs/infrastructure-as-code)

El despliegue instala las dependencias del servidor y descarga de Hugging Face los pesos ONNX del codificador. El catálogo SQLite contiene el corpus y las listas de postings comprimidas. El servidor lee por término y documento, y usa un índice FAISS ya generado. La ejecución mantiene un solo proceso y usa ONNX Runtime para reducir el consumo de memoria.

**Estado de publicación:** todavía no hay una URL pública asignada. Después de crear el servicio en tu cuenta de Render, copia la URL que Render genere aquí y en `informe_tecnico.pdf`; la configuración del Blueprint no equivale a un despliegue publicado.

## Modelos

El índice invertido y el preprocesamiento implementan los modelos léxicos; FAISS realiza la búsqueda vectorial de embeddings. No se usa un motor de búsqueda completo.

| Modelo | Puntuación |
|---|---|
| Binario | Jaccard sobre conjuntos de términos del vocabulario del corpus. |
| TF-IDF | Frecuencia cruda × log10(N/df) con similitud coseno. |
| BM25 | IDF positiva, k1 = 1.2 y b = 0.75. |
| Embeddings | MiniLM ajustado para recuperación en MS MARCO. Los embeddings de título y texto se generan con Sentence Transformers y se normalizan a norma L2. FAISS compara por producto interno y almacena los vectores en FP16. |

La web elige Snowball en inglés para el corpus financiero en inglés: reduce variantes morfológicas con poco coste y sin descargar un modelo lingüístico adicional. Filtra stopwords, conserva negaciones y aplica el mismo procesamiento a documentos y consultas. La CLI permite probar lematización como alternativa (`--modo lema`), pero cada ejecución elige un único método; no se aplican ambos juntos. La recuperación semántica usa el texto original, con los primeros 128 tokens por documento y consulta. El truncamiento reduce el costo de codificar y puede omitir información de los pasajes largos. El servicio calcula los embeddings de consulta con el mismo tokenizador, pesos ONNX y promedio enmascarado del codificador empleado para los documentos. [Ficha del modelo MiniLM](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L6-cos-v5)

## Evaluación

En **Evaluación**, elige el conjunto de prueba o validación y una consulta. La primera tabla muestra precisión y exhaustividad promedio en los primeros 10 resultados, y MAP sobre el ranking completo. La segunda muestra las métricas de la consulta seleccionada. **Comparar rankings** abre los cuatro resultados en paralelo y destaca los documentos con un juicio positivo. El conjunto `train` (5.500 consultas) no se mezcla con esas métricas; no se presenta como cero.

- **Precisión en los primeros 10** divide los aciertos entre 10. Las posiciones faltantes cuentan como fallos.
- **Exhaustividad (Recall) en los primeros 10** divide los documentos relevantes recuperados entre todos los documentos juzgados relevantes.
- **AP** suma la precisión en cada posición relevante del ranking completo y divide entre el total de relevantes. Los relevantes que el modelo no recupera aportan cero.
- **MAP** es la media de AP entre todas las consultas del conjunto. Precisión y exhaustividad usan el corte fijo de los primeros 10; AP y MAP usan el ranking completo devuelto por cada modelo. Cero en precisión o exhaustividad significa que no hubo relevantes en el Top 10; AP cero significa que no se recuperó ninguno en el ranking completo. No implica que falten datos.

### Resultados en prueba

Evaluación sobre 648 consultas oficiales.

| Modelo | Precisión media (Top 10) | Exhaustividad media (Top 10) | MAP |
|---|---:|---:|---:|
| Binario | 1,47 % | 5,89 % | 0,0372 |
| TF-IDF | 5,06 % | 21,37 % | 0,1344 |
| BM25 | 6,98 % | 31,11 % | 0,2077 |
| Embeddings | 6,14 % | 27,11 % | 0,1720 |

En la consulta 2076, sobre deducciones por gastos veterinarios en Canadá, embeddings obtuvo AP 1,0000 y BM25 0,0014. En la consulta 5125, sobre la adquisición de TEN por CBS en Australia, BM25 obtuvo AP 1,0000 y embeddings 0,0003. La segunda consulta identifica el tema con nombres propios que aportan coincidencias léxicas precisas.

Para reproducir los índices y la evaluación, instala `requirements-build.txt`, descarga las stopwords y ejecuta `python preparar.py todo` con el intérprete del entorno. También se aceptan las etapas `catalogo`, `embeddings` y `evaluacion`. La generación de embeddings puede tardar varios minutos en CPU. El informe técnico de entrega es `informe_tecnico.pdf`.

## Datos y atribución

Corpus y juicios de BEIR FiQA, distribuidos bajo CC BY-SA 4.0. El proyecto conserva los textos y los IDs, transforma la estructura a SQLite y genera índices para la búsqueda. Los datos publicados y sus transformaciones conservan CC BY-SA 4.0. [Corpus](https://huggingface.co/datasets/BeIR/fiqa) · [Juicios oficiales](https://huggingface.co/datasets/BeIR/fiqa-qrels) · [Licencia CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
