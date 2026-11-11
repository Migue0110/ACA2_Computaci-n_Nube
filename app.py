import time

import cohere
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

import rag_core as rc

# -----------------------------------------------------------------------------
# 1. Configuración de la página
# -----------------------------------------------------------------------------
st.set_page_config(page_title="IUAS - Asistente de Permanencia Estudiantil",
                   page_icon="🎓", layout="wide")

SALUDO = ("¡Hola! Soy tu asistente de permanencia IUAS. ¿En qué te puedo orientar hoy "
          "sobre el reglamento, las alertas tempranas o la homologación?")


# -----------------------------------------------------------------------------
# 2. Clientes (caché de recursos) y credenciales
# -----------------------------------------------------------------------------
@st.cache_resource
def iniciar_clientes():
    co_client = cohere.ClientV2(api_key=st.secrets["COHERE_API_KEY"])
    qdrant_client = QdrantClient(url=st.secrets["QDRANT_URL"], api_key=st.secrets["QDRANT_API_KEY"],
                                 timeout=60)
    g_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"],
                            http_options=types.HttpOptions(timeout=30_000))
    return co_client, qdrant_client, g_client


try:
    co_client, qdrant_client, g_client = iniciar_clientes()
except Exception as e:
    st.error(f"No se pudieron inicializar los servicios. Revisa los secrets de la app ({type(e).__name__}).")
    st.stop()

MODELO_LLM = st.secrets.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


def es_admin() -> bool:
    """Las operaciones de escritura exigen ADMIN_PASSWORD definido en los secrets."""
    clave = st.secrets.get("ADMIN_PASSWORD")
    if not clave:
        st.info("La edición está deshabilitada: define ADMIN_PASSWORD en los secrets de la app.")
        return False
    if st.session_state.get("admin_ok"):
        return True
    intento = st.text_input("Clave de administración", type="password", key="admin_clave")
    if intento and intento == clave:
        st.session_state.admin_ok = True
        return True
    if intento:
        st.error("Clave incorrecta.")
    return False


@st.cache_data(ttl=60, show_spinner=False)
def resumen_cacheado(_qdrant_client):
    return rc.resumen_indice(_qdrant_client)


def mostrar_fuentes(fuentes: list[dict]) -> None:
    if not fuentes:
        return
    with st.expander(f"🔍 Fuentes institucionales consultadas ({len(fuentes)})"):
        for i, f in enumerate(fuentes, 1):
            st.markdown(f"**[{i}] {f['etiqueta']}** · similitud `{f['score']:.4f}`")
            st.caption(f["extracto"] + "…")


# -----------------------------------------------------------------------------
# 3. Barra lateral
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎓 Asistente IUAS")
    vista = st.radio("Vista", ["💬 Chat", "📚 Gestión documental", "🧭 Índice vectorial"],
                     label_visibility="collapsed")
    st.divider()
    if vista == "💬 Chat":
        st.subheader("Parámetros de recuperación")
        ambito = st.selectbox("Ámbito de búsqueda", list(rc.AMBITOS),
                              help="Búsqueda híbrida: la similitud coseno ordena solo dentro del subconjunto filtrado.")
        desenlace = None
        if rc.AMBITOS[ambito] == "registro_estudiante":
            opcion = st.selectbox("Desenlace del estudiante", ["Cualquiera", "Dropout", "Enrolled", "Graduate"])
            desenlace = None if opcion == "Cualquiera" else opcion
        top_k = st.slider("Fragmentos recuperados (Top-k)", 1, 6, 3)
        umbral = st.slider("Umbral mínimo de similitud", 0.20, 0.80, rc.UMBRAL_SIMILITUD, 0.01,
                           help="Por debajo de este coseno Top-1 se responde que no hay información, sin llamar al LLM.")
        st.caption(f"LLM: `{MODELO_LLM}` · temperatura {rc.TEMPERATURA} · "
                   f"máx. {rc.MAX_TOKENS_SALIDA} tokens · memoria de {rc.TURNOS_MEMORIA} turnos")
        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": SALUDO}]
            st.rerun()


# -----------------------------------------------------------------------------
# 4. Vista de chat
# -----------------------------------------------------------------------------
def vista_chat():
    st.title("💬 Asistente Virtual de Permanencia")
    st.caption("Respuestas fundamentadas en la normativa IUAS indexada en Qdrant Cloud y generadas con Gemini.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": SALUDO}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])
            mostrar_fuentes(msg.get("fuentes", []))

    pregunta = st.chat_input("Escribe tu pregunta académica o administrativa aquí...")
    if not pregunta:
        return

    historial = st.session_state.messages[1:]  # sin el saludo inicial
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        t0 = time.time()
        fuentes, consulta = [], pregunta
        try:
            with st.spinner("Consultando normatividad institucional..."):
                consulta = rc.reformular_consulta(g_client, MODELO_LLM, historial, pregunta)
                filtro = rc.construir_filtro(rc.AMBITOS[ambito], target_raw=desenlace)
                puntos = rc.buscar(co_client, qdrant_client, consulta, top_k=top_k, filtro=filtro)
            fuentes = rc.resumir_fuentes(puntos)
            if not puntos or puntos[0].score < umbral:
                respuesta = rc.MENSAJE_SIN_INFO
                st.markdown(respuesta)
                fuentes = []
            else:
                salida = st.write_stream(rc.generar_respuesta(
                    g_client, MODELO_LLM, historial, pregunta, rc.construir_contexto(puntos)))
                respuesta = salida if isinstance(salida, str) else "".join(map(str, salida))
        except Exception as e:
            respuesta = ("No pude consultar la base de conocimiento en este momento "
                         f"({type(e).__name__}). Intenta de nuevo en unos segundos.")
            st.error(respuesta)

        partes = [f"⏱️ {time.time() - t0:.2f} s"]
        if consulta != pregunta:
            partes.append(f"consulta interpretada: «{consulta}»")
        if fuentes:
            partes.append(f"Top-1 {fuentes[0]['score']:.3f}")
        meta = " · ".join(partes)
        st.caption(meta)
        mostrar_fuentes(fuentes)

    st.session_state.messages.append({"role": "assistant", "content": respuesta,
                                      "fuentes": fuentes, "meta": meta})


# -----------------------------------------------------------------------------
# 5. Vista de gestión documental
# -----------------------------------------------------------------------------
def vista_gestion():
    st.title("📚 Gestión documental")
    st.caption("Documentos base del corpus normativo y registros indexados en la colección "
               f"`{rc.COLLECTION_NAME}`.")
    try:
        resumen = resumen_cacheado(qdrant_client)
    except Exception as e:
        st.error(f"No fue posible leer la colección en Qdrant Cloud ({type(e).__name__}).")
        return

    tabla = pd.DataFrame(resumen["documentos"]).rename(columns={
        "doc_id": "ID", "titulo": "Documento", "tipo": "Tipo", "puntos": "Fragmentos"})
    st.dataframe(tabla, hide_index=True, use_container_width=True)

    st.subheader("Cargar o actualizar un documento")
    st.caption("Formatos .md, .txt o .pdf. Si el nombre coincide con un documento existente, "
               "se reemplazan sus fragmentos (IDs deterministas, sin duplicados).")
    if not es_admin():
        return

    archivo = st.file_uploader("Documento", type=["md", "txt", "pdf"])
    if archivo is not None:
        texto = rc.texto_desde_archivo(archivo.name, archivo.getvalue())
        textos, payloads = rc.fragmentar(archivo.name, texto)
        st.write(f"Vista previa: **{len(textos)} fragmentos** para `{rc.normalizar_doc_id(archivo.name)}`.")
        if textos:
            with st.expander("Ver los primeros fragmentos"):
                for t in textos[:3]:
                    st.code(t[:600], language="markdown")
        if st.button("Vectorizar e indexar en Qdrant Cloud", type="primary", disabled=not textos):
            try:
                with st.spinner("Generando embeddings con Cohere e insertando en Qdrant..."):
                    n = rc.indexar_documento(co_client, qdrant_client, archivo.name, texto)
                resumen_cacheado.clear()
                st.success(f"Documento indexado: {n} fragmentos.")
            except Exception as e:
                st.error(f"Falló la indexación ({type(e).__name__}: {e}).")

    st.subheader("Eliminar un documento")
    ids = [d["doc_id"] for d in resumen["documentos"]]
    if ids:
        objetivo = st.selectbox("Documento a eliminar", ids,
                                format_func=lambda d: f"{rc.TITULOS.get(d, d)} ({d})")
        confirma = st.checkbox(f"Confirmo que deseo eliminar todos los puntos de `{objetivo}`")
        if st.button("Eliminar de la colección", disabled=not confirma):
            try:
                rc.eliminar_documento(qdrant_client, objetivo)
                resumen_cacheado.clear()
                st.success(f"Se eliminaron los puntos de `{objetivo}`.")
                st.rerun()
            except Exception as e:
                st.error(f"No se pudo eliminar ({type(e).__name__}).")


# -----------------------------------------------------------------------------
# 6. Vista del índice vectorial
# -----------------------------------------------------------------------------
def vista_indice():
    st.title("🧭 Índice vectorial")
    try:
        resumen = resumen_cacheado(qdrant_client)
    except Exception as e:
        st.error(f"No fue posible leer la colección en Qdrant Cloud ({type(e).__name__}).")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Puntos en la colección", resumen["puntos"])
    c2.metric("Estado", resumen["estado"])
    c3.metric("Dimensión", resumen["dimension"])
    c4.metric("Distancia", resumen["distancia"])
    st.caption("Índices de payload para búsqueda híbrida: " +
               ", ".join(f"`{i}`" for i in resumen["indices_payload"]))
    if st.button("Actualizar estado"):
        resumen_cacheado.clear()
        st.rerun()

    docs = pd.DataFrame(resumen["documentos"])
    if not docs.empty:
        st.subheader("Distribución de puntos por documento")
        st.bar_chart(docs.set_index("doc_id")["puntos"], horizontal=True)

        st.subheader("Explorar fragmentos y metadatos")
        doc = st.selectbox("Documento", docs["doc_id"].tolist(),
                           format_func=lambda d: f"{rc.TITULOS.get(d, d)} ({d})")
        try:
            filas = rc.fragmentos_de(qdrant_client, doc)
            df = pd.DataFrame(filas)
            st.dataframe(df[["orden", "seccion", "caracteres"]], hide_index=True, use_container_width=True)
            if filas:
                elegido = st.number_input("Ver texto completo del fragmento (orden)", min_value=0,
                                          max_value=max(int(f["orden"] or 0) for f in filas), value=0)
                texto = next((f["texto"] for f in filas if f["orden"] == elegido), None)
                if texto:
                    st.code(texto, language="markdown")
        except Exception as e:
            st.error(f"No se pudieron leer los fragmentos ({type(e).__name__}).")

    st.subheader("Búsqueda de prueba")
    consulta = st.text_input("Consulta", placeholder="¿Qué pasa si no pago la matrícula a tiempo?")
    k = st.slider("Top-k", 1, 10, 5, key="k_prueba")
    if consulta:
        try:
            puntos = rc.buscar(co_client, qdrant_client, consulta, top_k=k)
            st.dataframe(pd.DataFrame([{
                "rank": i, "similitud": round(p.score, 4),
                "supera umbral": p.score >= rc.UMBRAL_SIMILITUD,
                "fuente": rc.etiqueta_fuente(p.payload),
                "extracto": p.payload.get("texto_completo", "")[:160]} for i, p in enumerate(puntos, 1)]),
                hide_index=True, use_container_width=True)
        except Exception as e:
            st.error(f"La búsqueda falló ({type(e).__name__}).")


# -----------------------------------------------------------------------------
# 7. Enrutamiento
# -----------------------------------------------------------------------------
if vista == "💬 Chat":
    vista_chat()
elif vista == "📚 Gestión documental":
    vista_gestion()
else:
    vista_indice()
