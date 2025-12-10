"""
Asistente de Mantenimiento Técnico
"""

import streamlit as st
import os
import random
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

# ==================== CONFIGURACIÓN ====================
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
TEMPERATURE = 0.1

st.set_page_config(page_title="Asistente de Mantenimiento", page_icon="🏭", layout="wide")

# ==================== ESTÉTICA WEB =================



# ==================== HERRAMIENTAS (TOOLS) ====================

@tool
def consultar_manual_tecnico(pregunta: str) -> str:
    """
    Usa esta herramienta para buscar información técnica en el manual PDF.
    Útil para especificaciones, procedimientos y dudas sobre el equipo.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    docs = retriever.invoke(pregunta)
    
    contexto = ""
    for doc in docs:
        fuente = doc.metadata.get('source_file', 'Manual Desconocido')
        pagina = doc.metadata.get('page', '?')
        contexto += f"\n[Fuente: {fuente} - Pág {pagina}]\n{doc.page_content}\n"
        
        return contexto if contexto else "No encontré información relevante."


@tool
def crear_reporte_mantenimiento(equipo: str, tipo_falla: str, prioridad: str = "Media") -> str:
    """
    Genera un reporte de mantenimiento para equipos de seguridad electrónica.
    Útil cuando el usuario quiera reportar fallas en CCTV, alarmas, sensores, etc.
    """
    time.sleep(1)
    ticket_id = f"SEC-{random.randint(1000, 9999)}"  # Cambio de TKT a SEC
    fecha = time.strftime("%Y-%m-%d %H:%M")
    
    return f"""
✅ REPORTE DE INCIDENCIA CREADO
--------------------------------
Ticket ID: {ticket_id}
Fecha: {fecha}
Sistema/Equipo: {equipo}
Tipo de falla: {tipo_falla}
Prioridad: {prioridad}
Estado: Notificado al departamento de seguridad
Próximo paso: Técnico asignado en 2 horas (prioridad alta)
"""

@tool
def calcular_ancho_banda_cctv(camaras: int, resolucion: str = "1080p", fps: int = 25) -> str:
    """
    Calcula el ancho de banda necesario para un sistema CCTV.
    Útil para dimensionar redes de cámaras IP.
    
    Args:
        camaras: Número de cámaras
        resolucion: 720p, 1080p, 4K
        fps: Frames por segundo (típicamente 15-30)
    """
    # Bitrate aproximado por resolución (Mbps por cámara)
    bitrates = {
        "720p": 2,
        "1080p": 4,
        "4K": 15
    }
    
    bitrate_por_camara = bitrates.get(resolucion, 4)
    ancho_banda_total = camaras * bitrate_por_camara * (fps / 25)  # Normalizar a 25 fps
    
    return f"""
📊 CÁLCULO DE ANCHO DE BANDA
-----------------------------
Configuración:
- Número de cámaras: {camaras}
- Resolución: {resolucion}
- FPS: {fps}

Resultado:
- Bitrate por cámara: {bitrate_por_camara} Mbps
- Ancho de banda total requerido: {ancho_banda_total:.2f} Mbps
- Recomendación de switch: {ancho_banda_total * 1.3:.2f} Mbps (con 30% de margen)

💡 Consejo: Para grabación 24/7, considera espacio de almacenamiento de {ancho_banda_total * 0.45:.1f} TB/mes.
"""


# ==================== INICIALIZAR AGENTE ====================
@st.cache_resource
def inicializar_agente():
    # 1. Crear modelo LLM compatible con OpenAI (OpenRouter)
    model = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # 2. Lista de herramientas
    tools = [consultar_manual_tecnico, crear_reporte_mantenimiento, calcular_ancho_banda_cctv]
    
    # 3. Crear agente
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""Eres un Asistente Experto en Mantenimiento de Sistemas de Alarmas.
Tu objetivo es ayudar a técnicos de seguridad electrónica con problemas técnicos tanto en instalaciones como en fallos y errores de configuración.

HERRAMIENTAS DISPONIBLES:
- consultar_manual_tecnico: ÚSALA siempre que pregunten datos técnicos del manual.
- crear_reporte_mantenimiento: ÚSALA cuando quieran reportar fallas o averías.
- calcular_potencia_hidraulica: ÚSALA para cálculos con caudal y presión.

INSTRUCCIONES:
1. Si no sabes la respuesta, busca primero en el manual.
2. Si no está en el manual, dilo honestamente.
3. Responde siempre en español técnico y profesional."""
    )
    
    return agent

agent = inicializar_agente()

# ==================== INTERFAZ STREAMLIT ====================
st.title("Asistente Técnico")
st.markdown(f"**Motor:** Qwen3 Next 80B | **Estado:** Activo 🟢")
st.divider()

# Inicializar historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input del usuario
if input_usuario := st.chat_input("Escribe tu consulta o comando..."):
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": input_usuario})
    with st.chat_message("user"):
        st.markdown(input_usuario)
    
    # Generar respuesta del agente
    with st.chat_message("assistant"):
        contenedor = st.empty()
        contenedor.markdown("⚙️ *Procesando...*")
        
        try:
            # Invocar agente
            # El formato de entrada es {"messages": [...]}
            respuesta = agent.invoke({
                "messages": st.session_state.messages
            })
            
            # Extraer la respuesta del agente
            # En LangChain la respuesta viene en respuesta["messages"][-1]
            if "messages" in respuesta:
                output_texto = respuesta["messages"][-1].content
            else:
                output_texto = str(respuesta)
            
            contenedor.markdown(output_texto)
            st.session_state.messages.append({"role": "assistant", "content": output_texto})
            
        except Exception as e:
            contenedor.error(f"❌ Error del sistema: {str(e)}")
            st.code(str(e))  # Mostrar traceback para debugging
