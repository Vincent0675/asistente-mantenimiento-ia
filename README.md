# Asistente Inteligente de Mantenimiento en Seguridad Electrónica

Asistente conversacional para técnicos de seguridad electrónica, capaz de consultar manuales técnicos de varias marcas (AJAX, Risco, Hikvision, Hayward, etc.) y ejecutar acciones de mantenimiento de forma autónoma (creación de incidencias y cálculos de ancho de banda para CCTV).

---

## ✨ Funcionalidades principales

- **RAG (Retrieval-Augmented Generation):**  
    Consulta semántica sobre una colección de manuales PDF de sistemas de alarma, CCTV y otros dispositivos de seguridad electrónica, usando embeddings y ChromaDB.

- **Agente con herramientas (tools):**  
El LLM decide cuándo:
        - Buscar en los manuales (`consultar_manual_tecnico`)   
        - Crear un reporte de incidencia (`crear_reporte_mantenimiento`)   
        - Calcular el ancho de banda necesario para CCTV IP (`calcular_ancho_banda_cctv`)   

- **Interfaz web tipo chat:**  
    Desplegado con Streamlit, con historial de conversación persistente.

- **Monitorización con LangSmith:**  
    Trazas de uso (prompts, herramientas llamadas, tiempos de respuesta) para depuración y análisis.

---

## 🧱 Arquitectura (resumen rápido)

- **LLM:** `qwen/qwen3-next-80b-a3b-instruct` vía OpenRouter (optimizado para RAG y tool calling).
- **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (768 dimensiones, ejecutado en GPU si está disponible).
- **Vector Store:** ChromaDB persistente (`./chroma_db/`).
- **Framework:** LangChain 1.1.3 (`create_agent` + tools decoradas con `@tool`).
- **Interfaz:** Streamlit (`st.chat_input`, `st.chat_message`).
- **Monitorización:** LangSmith con endpoint europeo (`eu.api.smith.langchain.com`).

Los detalles completos están documentados en `DOCUMENTACION_TECNICA.md`.

---

## ⚙️ Instalación y ejecución

### 1. Clonar el repositorio

    git clone https://github.com/Vincent0675/asistente-mantenimiento-ia.git
    cd asistente-mantenimiento-ia

### 2. Crear entorno con Mamba/Conda

    mamba env create -f environment.yml
    mamba activate p4-asis


### 3. Configurar variables de entorno

Copia el ejemplo y edítalo con tus claves:

    cp .env.example .env

**Edita `.env` con:**

    OpenRouter

    OPENAI_API_KEY="sk-or-v1-TU_CLAVE_AQUI"
    OPENAI_API_BASE="https://openrouter.ai/api/v1"
    LangSmith (opcional pero recomendado)

    LANGSMITH_API_KEY="lsv2_pt_TU_CLAVE_LANGSMITH_AQUI"
    LANGSMITH_TRACING=true
    LANGSMITH_PROJECT="P4_Asistente_Mantenimiento"
    LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"


### 4. Añadir manuales técnicos

Coloca tus manuales PDF en la carpeta `data/` (no importa el nombre del archivo, mientras terminen en `.pdf`).

Ejemplos:
- `data/manual_hikvision_nvr.pdf`
- `data/manual_ajax_central.pdf`
- `data/manual_risco_alarmas.pdf`

### 5. Ingestar los manuales (crear la base vectorial)

Si ya existe una DB previa y quieres regenerarla

    rm -rf chroma_db/ # Elimina la DB previa

    python ingesta.py # Genera la nueva Base Vectorial


Esto:
- Lee todos los `.pdf` de `data/`
- Genera chunks de texto
- Crea embeddings con `all-mpnet-base-v2`
- Persiste todo en `./chroma_db/`

### 6. Lanzar la aplicación

    streamlit run app.py

La interfaz estará en:  
`http://localhost:8501`

---

## 🧪 Ejemplos de uso

Al abrir el chat, puedes probar consultas como:

- **Consulta RAG (manuales):**
  - `¿Cómo configuro la detección de movimiento en un NVR Hikvision?`
  - `¿Qué significan los códigos de error para el panel AJAX?`

- **Reporte de incidencias (agente):**
  - `Reporta que la cámara CAM-05 de la entrada principal no tiene visión nocturna, prioridad alta.`
  - `Genera un reporte por pérdida de comunicación con el panel Risco de la planta 2.`

- **Cálculo técnico (CCTV):**
  - `Calcula el ancho de banda para 12 cámaras a 1080p y 25 fps.`
  - `¿Cuánto ancho de banda necesito para 20 cámaras 4K a 15 fps?`

El agente decide automáticamente qué herramienta usar según la intención de la frase.

---

## 🧩 Estructura del repositorio   
```
asistente-mantenimiento-ia/
├── app.py # App Streamlit con el agente y las tools
├── ingesta.py # Script de ingesta multi-PDF → ChromaDB
├── environment.yml # Entorno reproducible (Mamba/Conda)
├── DOCUMENTACION_TECNICA.md # Documento técnico del proyecto
├── README.md # Este archivo
├── data/ # Manuales técnicos en PDF
│ ├── manual_hikvision_.pdf
│ ├── manual_ajax_.pdf
│ └── ...
└── chroma_db/ # Base de datos vectorial (se genera al ejecutar ingesta.py)
```

---

## 📊 Monitorización con LangSmith

Si `LANGSMITH_TRACING=true` y tu API key es válida:

- Cada interacción con el asistente se registra en LangSmith:
  - Prompt completo
  - Herramientas invocadas (`consultar_manual_tecnico`, `crear_reporte_mantenimiento`, `calcular_ancho_banda_cctv`)
  - Tiempos y tokens usados

Puedes consultar las trazas en tu proyecto `P4_Asistente_Mantenimiento` en:  
`https://eu.smith.langchain.com`

---
## 📚 Documentación adicional

- **Documento técnico completo:** `DOCUMENTACION_TECNICA.md`  
- Explica:
  - Técnicas de PLN usadas (embeddings, RAG, agentes)
  - Justificación de su utilidad en mantenimiento de seguridad electrónica
  - Diagrama conversacional en Mermaid
  - Detalles de modelos, hiperparámetros y arquitectura

---
## 👤 Autor

**Nombre:** Byron Vincent Blatch Rodríguez  
**Curso:** Especialización en Big Data e Inteligencia Artificial   
**Asignatura:** Modelos de Inteligencia Artificial   

---
## 📄 Licencia

- Código: Apache 2.0  
- Documentación: Creative Commons BY-SA 4.0