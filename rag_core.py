"""
Núcleo del asistente RAG de permanencia estudiantil (IUAS).

Reúne la lógica que no depende de la interfaz: recuperación semántica e
híbrida en Qdrant Cloud, generación con Gemini (SDK google-genai), memoria
conversacional, fragmentación de documentos e indexación. La app de Streamlit
(app.py) importa estas funciones; el notebook del ACA usa la misma
estrategia de fragmentación, de modo que un documento indexado desde la app
produce los mismos fragmentos e identificadores que en Colab.
"""

from __future__ import annotations

import io
import re
import time
import uuid
from typing import Iterable, Iterator

import yaml
from google.genai import types
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import models

# -----------------------------------------------------------------------------
# Parámetros del sistema
# -----------------------------------------------------------------------------
COLLECTION_NAME = "edu_rag_permanencia"
EMBED_MODEL = "embed-multilingual-v3.0"
VECTOR_SIZE = 1024
CHUNK_SIZE, CHUNK_OVERLAP = 700, 100
BATCH_SIZE, MAX_REINTENTOS = 32, 4

# En la evaluación del notebook, la consulta de control fuera de dominio obtuvo
# un coseno Top-1 de 0,347 y ninguna consulta del dominio bajó de 0,57.
# Un umbral de 0,45 separa ambos grupos y evita llamar al LLM sin evidencia.
UMBRAL_SIMILITUD = 0.45

TEMPERATURA = 0.1          # respuestas deterministas y pegadas al contexto
MAX_TOKENS_SALIDA = 800    # 400 truncaba respuestas largas (homologación, víctimas)
TURNOS_MEMORIA = 3         # pares pregunta-respuesta que se envían como historial

MENSAJE_SIN_INFO = (
    "Lo siento, la información solicitada no se encuentra disponible "
    "en la normativa institucional cargada."
)

TITULOS = {
    "01_reglamento_academico_estudiantil": "Reglamento Académico Estudiantil",
    "02_protocolo_acompanamiento_psicopedagogico": "Protocolo de Acompañamiento Psicopedagógico",
    "03_guia_homologacion_reconocimiento_creditos": "Guía de Homologación y Reconocimiento de Créditos",
    "registro_estudiante": "Registros de estudiantes (dataset UCI)",
}

AMBITOS = {
    "Toda la base": None,
    "Solo normativa": "documento_normativo",
    "Solo registros de estudiantes": "registro_estudiante",
}

SYSTEM_PROMPT = f"""Eres el Asistente Virtual Oficial de Permanencia Estudiantil de la Institución Universitaria Andina del Saber (IUAS).

Reglas obligatorias:
1. Responde ÚNICAMENTE con la información del CONTEXTO RECUPERADO que acompaña la última pregunta. El historial sirve solo para entender a qué se refiere el estudiante; no es fuente.
2. Si el contexto no contiene la respuesta, responde exactamente: "{MENSAJE_SIN_INFO}" y no agregues fuentes.
3. No inventes cifras, plazos, requisitos ni nombres de dependencias.
4. Cita cada afirmación normativa con el formato [Fuente: nombre de la sección], usando el nombre que aparece en la cabecera de cada fuente.
5. Si el contexto incluye registros de estudiantes, preséntalos como casos de referencia anonimizados, nunca como datos de la persona que pregunta.
6. Responde en español, con tono claro, empático y profesional; usa listas breves cuando ayuden."""


# -----------------------------------------------------------------------------
# Embeddings (Cohere)
# -----------------------------------------------------------------------------
def _vectores(respuesta) -> list[list[float]]:
    """Extrae los vectores float; el SDK v5 expone `.float` y el v7 `.float_`."""
    emb = respuesta.embeddings
    vectores = getattr(emb, "float_", None)
    if vectores is None:
        vectores = getattr(emb, "float")
    return vectores


def embeber(co_client, textos: list[str], input_type: str) -> list[list[float]]:
    """Vectoriza en lotes de 32 con reintentos y espera exponencial."""
    salida: list[list[float]] = []
    for i in range(0, len(textos), BATCH_SIZE):
        lote = textos[i:i + BATCH_SIZE]
        for intento in range(MAX_REINTENTOS):
            try:
                res = co_client.embed(texts=lote, model=EMBED_MODEL,
                                      input_type=input_type, embedding_types=["float"])
                salida.extend(_vectores(res))
                break
            except Exception:
                if intento == MAX_REINTENTOS - 1:
                    raise
                time.sleep(2 ** (intento + 1))
    return salida


# -----------------------------------------------------------------------------
# Recuperación (semántica + filtro de payload)
# -----------------------------------------------------------------------------
def construir_filtro(tipo: str | None = None, **condiciones) -> models.Filter | None:
    must = []
    if tipo:
        must.append(models.FieldCondition(key="tipo", match=models.MatchValue(value=tipo)))
    for clave, valor in condiciones.items():
        if valor is not None:
            must.append(models.FieldCondition(key=clave, match=models.MatchValue(value=valor)))
    return models.Filter(must=must) if must else None


def buscar(co_client, qdrant_client, consulta: str, top_k: int = 3,
           filtro: models.Filter | None = None):
    vector = embeber(co_client, [consulta], "search_query")[0]
    return qdrant_client.query_points(
        collection_name=COLLECTION_NAME, query=vector, limit=top_k,
        query_filter=filtro, with_payload=True,
    ).points


def etiqueta_fuente(payload: dict) -> str:
    """Nombre legible de la fuente: título del documento en lugar del doc_id."""
    if payload.get("tipo") == "registro_estudiante":
        return f"Registro de estudiante UCI · ID {payload.get('id_original', '¿?')}"
    doc_id = payload.get("doc_id", "")
    titulo = TITULOS.get(doc_id) or payload.get("titulo") or doc_id or "Normativa"
    seccion = payload.get("seccion", "")
    partes = [p.strip() for p in seccion.split(">")] if seccion else []
    if partes and partes[0] in (doc_id, payload.get("titulo")):
        partes = partes[1:]
    return " › ".join([titulo] + partes)


def construir_contexto(puntos) -> str:
    bloques = []
    for i, p in enumerate(puntos, 1):
        bloques.append(f"--- [FUENTE {i}: {etiqueta_fuente(p.payload)}] ---\n"
                       f"{p.payload.get('texto_completo', '')}")
    return "\n\n".join(bloques)


def resumir_fuentes(puntos) -> list[dict]:
    """Versión serializable de las fuentes para guardar en el historial."""
    return [{"etiqueta": etiqueta_fuente(p.payload), "score": float(p.score),
             "extracto": p.payload.get("texto_completo", "")[:280]} for p in puntos]


# -----------------------------------------------------------------------------
# Memoria conversacional y generación (Gemini, SDK google-genai)
# -----------------------------------------------------------------------------
def _contenido(rol: str, texto: str) -> types.Content:
    return types.Content(role=rol, parts=[types.Part(text=texto)])


def historial_a_contents(historial: Iterable[dict], turnos: int = TURNOS_MEMORIA) -> list[types.Content]:
    """Convierte los últimos turnos (sin el contexto recuperado) al formato de Gemini."""
    mensajes = [m for m in historial if m.get("role") in ("user", "assistant") and m.get("content")]
    mensajes = mensajes[-2 * turnos:]
    return [_contenido("user" if m["role"] == "user" else "model", m["content"]) for m in mensajes]


def reformular_consulta(g_client, modelo: str, historial: list[dict], pregunta: str) -> str:
    """Convierte una pregunta de seguimiento en una consulta autónoma para el retrieval.

    Ejemplo: tras preguntar por el periodo de prueba, «¿y si estudio de noche?»
    se reescribe como «periodo de prueba para estudiantes de jornada nocturna».
    Si falla o no hay historial, devuelve la pregunta original.
    """
    previos = historial_a_contents(historial)
    if not previos:
        return pregunta
    instruccion = ("Reescribe la ÚLTIMA pregunta del estudiante como una consulta de búsqueda "
                   "autónoma en español, resolviendo pronombres y referencias con el historial. "
                   "Devuelve solo la consulta, sin comillas ni explicación.")
    try:
        r = g_client.models.generate_content(
            model=modelo,
            contents=previos + [_contenido("user", pregunta)],
            config=types.GenerateContentConfig(system_instruction=instruccion,
                                               temperature=0.0, max_output_tokens=80),
        )
        texto = (r.text or "").strip().strip('"«»')
        return texto or pregunta
    except Exception:
        return pregunta


def generar_respuesta(g_client, modelo: str, historial: list[dict], pregunta: str,
                      contexto: str) -> Iterator[str]:
    """Genera la respuesta en streaming; ante un error entrega un aviso en lugar de romper la app."""
    contents = historial_a_contents(historial) + [_contenido(
        "user", f"CONTEXTO RECUPERADO DE QDRANT CLOUD:\n{contexto}\n\nPREGUNTA DEL ESTUDIANTE:\n{pregunta}")]
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=TEMPERATURA,
                                         max_output_tokens=MAX_TOKENS_SALIDA)
    try:
        hubo_texto = False
        for chunk in g_client.models.generate_content_stream(model=modelo, contents=contents, config=config):
            texto = getattr(chunk, "text", None)
            if texto:
                hubo_texto = True
                yield texto
        if not hubo_texto:
            yield ("El modelo no devolvió texto para esta consulta (posible filtro de seguridad). "
                   "Reformula la pregunta o consulta a la Coordinación de Permanencia.")
    except Exception as e:  # cuota, timeout o red
        yield f"\n\n⚠️ No fue posible completar la respuesta con Gemini ({type(e).__name__}). Intenta de nuevo."


# -----------------------------------------------------------------------------
# Fragmentación (misma estrategia que la Fase 4 del notebook)
# -----------------------------------------------------------------------------
_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " ", ""])


def normalizar_doc_id(nombre_archivo: str) -> str:
    base = re.sub(r"\.(md|markdown|txt|pdf)$", "", nombre_archivo, flags=re.I)
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", base).strip("_").lower() or "documento"


def separar_frontmatter(texto: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n", texto, re.S)
    if not m:
        return {}, texto
    try:
        return (yaml.safe_load(m.group(1)) or {}), texto[m.end():]
    except yaml.YAMLError:
        return {}, texto[m.end():]


def texto_desde_archivo(nombre: str, contenido: bytes) -> str:
    if nombre.lower().endswith(".pdf"):
        from pypdf import PdfReader
        lector = PdfReader(io.BytesIO(contenido))
        return "\n\n".join((pagina.extract_text() or "") for pagina in lector.pages)
    return contenido.decode("utf-8", errors="replace")


def fragmentar(nombre_archivo: str, texto: str) -> tuple[list[str], list[dict]]:
    """Devuelve (textos a vectorizar, payloads) con encabezado contextual por fragmento."""
    doc_id = normalizar_doc_id(nombre_archivo)
    meta_doc, cuerpo = separar_frontmatter(texto)
    titulo = meta_doc.get("titulo", doc_id)
    es_markdown = nombre_archivo.lower().endswith((".md", ".markdown"))
    secciones = _md_splitter.split_text(cuerpo) if es_markdown else None

    textos, payloads = [], []
    unidades = ([(" > ".join([titulo] + [s.metadata[h] for h in ("h2", "h3") if h in s.metadata]),
                  s.page_content) for s in secciones] if secciones is not None else [(titulo, cuerpo)])
    idx = 0
    for ruta, contenido in unidades:
        for trozo in _char_splitter.split_text(contenido):
            texto_chunk = f"[{ruta}]\n{trozo}"
            textos.append(texto_chunk)
            payloads.append({
                "tipo": "documento_normativo", "doc_id": doc_id, "fuente": nombre_archivo,
                "titulo": titulo, "tipo_documento": meta_doc.get("tipo_documento"),
                "seccion": ruta, "chunk_index": idx,
                "variables_dataset": meta_doc.get("variables_dataset_relacionadas", []),
                "texto_completo": texto_chunk,
            })
            idx += 1
    return textos, payloads


def id_punto(meta: dict) -> str:
    """UUID determinista idéntico al del notebook: re-indexar actualiza, no duplica."""
    clave = f"{meta['tipo']}|{meta['doc_id']}|{meta.get('chunk_index', meta.get('id_original'))}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, clave))


# -----------------------------------------------------------------------------
# Gestión de la colección
# -----------------------------------------------------------------------------
def eliminar_documento(qdrant_client, doc_id: str) -> None:
    qdrant_client.delete(
        collection_name=COLLECTION_NAME, wait=True,
        points_selector=models.FilterSelector(filter=construir_filtro(doc_id=doc_id)),
    )


def indexar_documento(co_client, qdrant_client, nombre_archivo: str, texto: str) -> int:
    """Fragmenta, vectoriza e inserta un documento; reemplaza la versión previa del mismo doc_id."""
    textos, payloads = fragmentar(nombre_archivo, texto)
    if not textos:
        return 0
    vectores = embeber(co_client, textos, "search_document")
    eliminar_documento(qdrant_client, payloads[0]["doc_id"])
    for i in range(0, len(textos), BATCH_SIZE):
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME, wait=True,
            points=[models.PointStruct(id=id_punto(m), vector=v, payload=m)
                    for v, m in zip(vectores[i:i + BATCH_SIZE], payloads[i:i + BATCH_SIZE])],
        )
    return len(textos)


def resumen_indice(qdrant_client) -> dict:
    """Estado de la colección y conteo de puntos por documento (vía scroll paginado)."""
    info = qdrant_client.get_collection(COLLECTION_NAME)
    vectores = info.config.params.vectors
    conteo: dict[str, dict] = {}
    offset = None
    while True:
        puntos, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME, limit=256, offset=offset,
            with_payload=["doc_id", "tipo", "fuente"], with_vectors=False)
        for p in puntos:
            doc_id = p.payload.get("doc_id", "sin_doc_id")
            fila = conteo.setdefault(doc_id, {"doc_id": doc_id, "titulo": TITULOS.get(doc_id, doc_id),
                                              "tipo": p.payload.get("tipo", ""), "puntos": 0})
            fila["puntos"] += 1
        if offset is None:
            break
    return {
        "puntos": info.points_count,
        "estado": str(getattr(info.status, "value", info.status)),
        "dimension": getattr(vectores, "size", None),
        "distancia": str(getattr(getattr(vectores, "distance", None), "value", "")),
        "indices_payload": sorted((info.payload_schema or {}).keys()),
        "documentos": sorted(conteo.values(), key=lambda f: (f["tipo"] != "documento_normativo", f["doc_id"])),
    }


def fragmentos_de(qdrant_client, doc_id: str, limite: int = 500) -> list[dict]:
    puntos, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME, scroll_filter=construir_filtro(doc_id=doc_id),
        limit=limite, with_payload=True, with_vectors=False)
    filas = [{"orden": p.payload.get("chunk_index", p.payload.get("id_original")),
              "seccion": etiqueta_fuente(p.payload),
              "caracteres": len(p.payload.get("texto_completo", "")),
              "texto": p.payload.get("texto_completo", "")} for p in puntos]
    return sorted(filas, key=lambda f: (f["orden"] is None, f["orden"]))
