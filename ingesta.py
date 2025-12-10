"""
Script de Ingesta Multi-Documento
Proyecto 4: Asistente de Seguridad Electrónica

Mejoras:
- Procesa TODOS los .pdf en la carpeta data/
- Usa GPU para embeddings (all-mpnet-base-v2)
- Añade metadatos del archivo origen (para saber de qué marca es la respuesta)
"""

import os
import glob
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ==================== CONFIGURACIÓN ====================
DATA_DIR = "data"
DB_PATH = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ingestar_documentos():
    print("=" * 70)
    print(f"🏭 INICIANDO INGESTA MULTI-DOCUMENTO ({DEVICE.upper()})")
    print("=" * 70)
    
    # 1. Buscar todos los PDFs en la carpeta data/
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"❌ ERROR: No se encontraron archivos PDF en '{DATA_DIR}/'")
        return

    print(f"📚 Encontrados {len(pdf_files)} manuales:")
    for f in pdf_files:
        print(f"   - {os.path.basename(f)}")
    
    # 2. Cargar documentos
    all_documents = []
    print("\n📄 Cargando documentos...")
    
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # Añadir metadata extra (nombre del archivo)
            for doc in docs:
                doc.metadata["source_file"] = os.path.basename(pdf_path)
            
            all_documents.extend(docs)
            print(f"   ✓ {os.path.basename(pdf_path)}: {len(docs)} páginas")
        except Exception as e:
            print(f"   ⚠️ Error cargando {pdf_path}: {e}")

    print(f"   Total páginas cargadas: {len(all_documents)}")
    
    # 3. Dividir en Chunks
    print(f"\n🧩 Dividiendo texto en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"   ✓ Generados {len(chunks)} fragmentos totales")
    
    # 4. Embeddings + Guardado
    print(f"\n🧠 Generando embeddings y guardando en ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("\n" + "=" * 70)
    print("✅ INGESTA COMPLETADA")
    print(f"   - Base de datos: ./{DB_PATH}/")
    print(f"   - Total vectores: {len(chunks)}")
    print("=" * 70)

if __name__ == "__main__":
    ingestar_documentos()
