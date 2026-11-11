# Asistente RAG de permanencia estudiantil · IUAS

ACA 2 · Computación en la Nube · Especialización en Inteligencia Artificial (CUN) · Grupo 3
Docente: Dr. Hamilton Rivera Flor · Pilar: Educación

**Integrantes:** Johnny Alirio Blanquicett Cardona, Miguel Orlando Mora Moreno, Ramón Benjamín Buelvas Sierra y Rubby Yesenia Parrado Rodríguez.

- Aplicación desplegada: https://aca2computaci-nnube-4nrzzmjcrh8l2kkeelipgw.streamlit.app/
- Video de sustentación: _[agregar enlace]_

## Qué hace

El sistema responde preguntas de estudiantes y tutores sobre la normativa de permanencia de la Institución Universitaria Andina del Saber (IUAS), una institución ficticia cuyo corpus se alineó con el dataset UCI *Predict Students' Dropout and Academic Success* (Martins et al., 2021). Cada respuesta se fundamenta en fragmentos recuperados de Qdrant Cloud y cita la sección normativa de origen; cuando la evidencia no alcanza el umbral de similitud, el asistente declara que la información no está disponible en lugar de generar una respuesta.

## Arquitectura

```
Indexación (Colab, notebook)           Consulta (Streamlit Cloud, app.py)
3 normas .md + 51 perfiles UCI         Pregunta + historial
        │                                     │  reformulación (Gemini)
Chunking Markdown + 700/100 car.              ▼
        │                              Cohere embed v3 (search_query)
Cohere embed v3 (search_document)             │
        └──────────►  Qdrant Cloud  ◄─────────┘  Top-k + filtro de payload
                    173 vectores · 1024 d · coseno
                                              │  umbral de similitud
                                              ▼
                                  Prompt RAG + Gemini (T = 0,1, streaming)
                                              │
                                   Respuesta con [Fuente: sección]
```

## Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `ACA2_Computacion_en_la_nube_G3.ipynb` | Ingesta, EDA, chunking, indexación, evaluación del retrieval y pruebas del pipeline RAG |
| `app.py` | Interfaz Streamlit: chat, gestión documental e índice vectorial |
| `rag_core.py` | Lógica RAG reutilizable: recuperación híbrida, memoria, generación, fragmentación e indexación |
| `requirements.txt` | Dependencias |
| `.streamlit/secrets.toml.example` | Plantilla de credenciales |

## Vistas de la aplicación

1. **Chat:** historial conversacional con memoria de tres turnos, reformulación de preguntas de seguimiento, respuesta en streaming, fuentes con puntaje de similitud, ámbito de búsqueda (toda la base, solo normativa o solo registros) y umbral ajustable.
2. **Gestión documental:** listado de documentos indexados, carga de archivos `.md`, `.txt` o `.pdf` con vista previa de fragmentos, re-indexación sin duplicados y eliminación por documento. Requiere `ADMIN_PASSWORD`.
3. **Índice vectorial:** estado de la colección, dimensión y métrica, índices de payload, distribución de puntos por documento, exploración de fragmentos y búsqueda de prueba con puntajes.

## Ejecución local

```bash
git clone https://github.com/Migue0110/ACA2_Computaci-n_Nube.git
cd ACA2_Computaci-n_Nube
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # completar las llaves
streamlit run app.py
```

La colección `edu_rag_permanencia` debe existir: se crea y se puebla ejecutando el notebook (Fases 3 a 5).

## Despliegue en Streamlit Cloud

1. Conectar el repositorio y seleccionar `app.py` como archivo principal.
2. En *App settings › Secrets* pegar el contenido de `secrets.toml.example` con las llaves reales.

## Gestión de credenciales

Las llaves viven en Colab Secrets (notebook) y en `st.secrets` (app); el `.gitignore` excluye `.env` y `.streamlit/secrets.toml`. Ninguna llave aparece en el código ni en las salidas.

## Hiperparámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| Temperatura | 0,1 | Respuestas deterministas y pegadas al contexto |
| Máx. tokens de salida | 800 | Las respuestas de homologación superan 400 tokens |
| Top-k | 3 (ajustable) | Hit@3 = 90 % en la evaluación del notebook |
| Umbral de similitud | 0,45 | Control fuera de dominio = 0,347; consultas del dominio ≥ 0,57 |
| Chunk / solapamiento | 700 / 100 caracteres | Un artículo por vector, sin cortar oraciones |

## Referencias

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474. https://arxiv.org/abs/2005.11401

Martins, M. V., Real, M. B., Machado, J., & Becker, A. (2021). *Predict students' dropout and academic success* [Conjunto de datos]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
