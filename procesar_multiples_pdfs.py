import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import glob

def extract_text(pdf_path):
    """Extrae texto de un archivo PDF"""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text() or ""
            full_text += text + "\n"
        return full_text
    except Exception as e:
        print(f"Error al procesar {pdf_path}: {e}")
        return ""

def split_chunks(text, chunk_size=500, overlap=50):
    """Divide el texto en fragmentos"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

def procesar_pdfs_en_carpeta(carpeta_pdfs, chunk_size=500, overlap=50):
    """Procesa todos los PDFs en una carpeta"""
    
    # Buscar todos los archivos PDF en la carpeta
    pdfs = glob.glob(os.path.join(carpeta_pdfs, "*.pdf"))
    
    if not pdfs:
        print(f"No se encontraron archivos PDF en {carpeta_pdfs}")
        return [], []
    
    todos_fragments = []
    metadatos = []  # Para guardar información de cada fragmento
    
    print(f"Procesando {len(pdfs)} archivos PDF...")
    
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\nProcesando {i}/{len(pdfs)}: {os.path.basename(pdf_path)}")
        
        # Extraer texto
        texto = extract_text(pdf_path)
        if not texto.strip():
            print(f"  ⚠️  No se pudo extraer texto de {os.path.basename(pdf_path)}")
            continue
        
        # Dividir en fragmentos
        fragments = split_chunks(texto, chunk_size, overlap)
        
        # Agregar metadatos para cada fragmento
        for j, fragment in enumerate(fragments):
            metadatos.append({
                'archivo': os.path.basename(pdf_path),
                'fragmento_numero': j + 1,
                'total_fragments_archivo': len(fragments)
            })
        
        todos_fragments.extend(fragments)
        
        # Guardar fragmentos individuales
        nombre_base = os.path.splitext(os.path.basename(pdf_path))[0]
        for j, fragment in enumerate(fragments, 1):
            nombre_archivo = f"fragmentos/{nombre_base}_fragmento_{j}.txt"
            os.makedirs("fragmentos", exist_ok=True)
            with open(nombre_archivo, "w", encoding="utf-8") as f:
                f.write(fragment)
        
        print(f"  ✅ {len(fragments)} fragmentos generados")
    
    return todos_fragments, metadatos

def generar_embeddings_y_faiss(fragments, metadatos):
    """Genera embeddings y crea índice FAISS"""
    
    if not fragments:
        print("No hay fragmentos para procesar")
        return None, None
    
    print(f"\nGenerando embeddings para {len(fragments)} fragmentos...")
    
    # Cargar modelo
    modelo = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generar embeddings
    embeddings = modelo.encode(fragments, show_progress_bar=True)
    
    # Crear índice FAISS
    print("Creando índice FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    
    # Guardar todo
    faiss.write_index(index, "faiss_index_multiple.idx")
    np.save('embeddings_multiple.npy', embeddings)
    
    # Guardar metadatos
    import json
    with open('metadatos_fragments.json', 'w', encoding='utf-8') as f:
        json.dump(metadatos, f, ensure_ascii=False, indent=2)
    
    # Guardar fragmentos
    with open('fragments_multiple.txt', 'w', encoding='utf-8') as f:
        for i, fragment in enumerate(fragments):
            f.write(f"=== FRAGMENTO {i+1} ===\n")
            f.write(f"Archivo: {metadatos[i]['archivo']}\n")
            f.write(f"Fragmento: {metadatos[i]['fragmento_numero']}\n")
            f.write(fragment + "\n\n")
    
    print("✅ Índice FAISS y archivos guardados:")
    print("  - faiss_index_multiple.idx")
    print("  - embeddings_multiple.npy")
    print("  - metadatos_fragments.json")
    print("  - fragments_multiple.txt")
    
    return index, embeddings

def buscar_fragmentos_relevantes(pregunta, index, fragments, modelo, top_k=3):
    """Busca fragmentos relevantes para una pregunta"""
    pregunta_embedding = modelo.encode([pregunta])
    _, indices = index.search(pregunta_embedding, top_k)
    
    fragmentos_relevantes = []
    for idx in indices[0]:
        fragmentos_relevantes.append(fragments[idx])
    
    return fragmentos_relevantes

# Ejemplo de uso
if __name__ == "__main__":
    # Especifica la carpeta donde están tus PDFs
    carpeta_pdfs = r"C:\Users\crist\Desktop\tesisna\fuentes\tema"
    
    # Procesar todos los PDFs
    fragments, metadatos = procesar_pdfs_en_carpeta(carpeta_pdfs)
    
    if fragments:
        # Generar embeddings y FAISS
        index, embeddings = generar_embeddings_y_faiss(fragments, metadatos)
        
        # Ejemplo de búsqueda
        if index is not None:
            modelo = SentenceTransformer('all-MiniLM-L6-v2')
            pregunta = "¿Cuál es el impacto de las TIC en la agricultura?"
            
            print(f"\n=== Ejemplo de búsqueda: '{pregunta}' ===")
            resultados = buscar_fragmentos_relevantes(pregunta, index, fragments, modelo, top_k=2)
            
            for i, resultado in enumerate(resultados, 1):
                print(f"\n--- Resultado {i} ---")
                print(resultado[:300] + "..." if len(resultado) > 300 else resultado)
    else:
        print("No se pudieron procesar archivos PDF") 