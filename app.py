import streamlit as st
import cohere
from qdrant_client import QdrantClient
import google.generativeai as genai
import time

# -----------------------------------------------------------------------------
# 1. Configuración de la Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IUAS - Asistente de Permanencia Estudiantil",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Asistente Virtual IUAS")
st.caption("Sistema RAG para la orientación y retención de estudiantes")

# -----------------------------------------------------------------------------
# 2. Inicialización de Clientes y Conexiones (Con Caché de Recursos)
# -----------------------------------------------------------------------------
@st.cache_resource
def iniciar_clientes():
    # Carga de credenciales desde st.secrets
    cohere_key = st.secrets["COHERE_API_KEY"]
    qdrant_url = st.secrets["QDRANT_URL"]
    qdrant_key = st.secrets["QDRANT_API_KEY"]
    gemini_key = st.secrets["GEMINI_API_KEY"]

    co_client = cohere.ClientV2(api_key=cohere_key)
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    genai.configure(api_key=gemini_key)
    
    # Modelo verificado
    llm_model = genai.GenerativeModel('gemini-3.5-flash-lite')
    
    return co_client, qdrant_client, llm_model

try:
    co_client, qdrant_client, llm_model = iniciar_clientes()
    COLLECTION_NAME = "iuas_estudiantes_rag"
    EMBED_MODEL = "embed-multilingual-v3.0"
except Exception as e:
    st.error(f"Error al conectar con los servicios API: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Lógica del Pipeline RAG
# -----------------------------------------------------------------------------
def obtener_embedding(texto: str):
    response = co_client.embed(
        texts=[texto],
        model=EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"]
    )
    return response.embeddings.float[0]

def buscar_fuentes(pregunta: str, top_k=3):
    vector = obtener_embedding(pregunta)
    puntos = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True
    ).points
    return puntos

def generar_respuesta_rag(pregunta: str, puntos):
    contexto_str = ""
    for idx, p in enumerate(puntos, 1):
        payload = p.payload
        seccion = payload.get("seccion", payload.get("tipo", "Normativa"))
        texto = payload.get("texto_completo", "")
        contexto_str += f"\n--- [FUENTE {idx}: {seccion}] ---\n{texto}\n"

    prompt_sistema = f"""
Eres el Asistente Virtual Oficial de Permanencia Estudiantil de la Institución Universitaria Andina del Saber (IUAS).
Responde la pregunta del estudiante basándote ÚNICAMENTE en el CONTEXTO RECUPERADO.
Si la información no está disponible en el contexto, responde exactamente: 
"Lo siento, la información solicitada no se encuentra disponible en la normativa institucional cargada."

Reglas:
1. Responde de forma clara, empática y profesional.
2. Cita siempre la sección o norma utilizada usando el formato [Fuente: Nombre de la Sección].

CONTEXTO RECUPERADO DE QDRANT CLOUD:
{contexto_str}

PREGUNTA DEL ESTUDIANTE:
{pregunta}
"""

    config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=400)
    
    # Streaming de respuesta para UX fluida
    respuesta_stream = llm_model.generate_content(
        prompt_sistema,
        generation_config=config,
        stream=True,
        request_options={"timeout": 15.0}
    )
    return respuesta_stream

# -----------------------------------------------------------------------------
# 4. Interfaz de Chat (Historial de Mensajes)
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu asistente de permanencia IUAS. ¿En qué te puedo orientar hoy sobre el reglamento o apoyos institucionales?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "fuentes" in msg and msg["fuentes"]:
            with st.expander("🔍 Ver fuentes institucionales consultadas"):
                for idx, f in enumerate(msg["fuentes"], 1):
                    sec = f.payload.get('seccion', f.payload.get('tipo', 'Norma'))
                    st.write(f"**[{idx}] {sec}** (Similitud: `{f.score:.4f}`)")
                    st.caption(f.payload.get('texto_completo', '')[:200] + "...")

# Captura de la consulta del usuario
if pregunta_user := st.chat_input("Escribe tu pregunta académica o administrativa aquí..."):
    
    st.session_state.messages.append({"role": "user", "content": pregunta_user})
    with st.chat_message("user"):
        st.markdown(pregunta_user)

    with st.chat_message("assistant"):
        with st.spinner("Consultando normatividad institucional..."):
            t0 = time.time()
            fuentes = buscar_fuentes(pregunta_user, top_k=3)
            stream_respuesta = generar_respuesta_rag(pregunta_user, fuentes)
            
            # Renderizado en tiempo real (palabra por palabra)
            respuesta_completa = st.write_stream(stream_respuesta)
            t_total = round(time.time() - t0, 2)
            
            st.caption(f"⏱️ Tiempo de respuesta: {t_total}s")
            
            if fuentes:
                with st.expander("🔍 Ver fuentes institucionales consultadas"):
                    for idx, f in enumerate(fuentes, 1):
                        sec = f.payload.get('seccion', f.payload.get('tipo', 'Norma'))
                        st.write(f"**[{idx}] {sec}** (Similitud: `{f.score:.4f}`)")
                        st.caption(f.payload.get('texto_completo', '')[:200] + "...")

    # Guardar en el historial de la sesión
    st.session_state.messages.append({
        "role": "assistant",
        "content": respuesta_completa,
        "fuentes": fuentes
    })