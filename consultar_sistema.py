import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

def cargar_sistema():
    """Carga el sistema de búsqueda semántica"""
    try:
        # Cargar índice FAISS
        index = faiss.read_index("faiss_index_multiple.idx")
        
        # Cargar embeddings
        embeddings = np.load('embeddings_multiple.npy')
        
        # Cargar fragmentos
        with open('fragments_multiple.txt', 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        # Cargar metadatos
        with open('metadatos_fragments.json', 'r', encoding='utf-8') as f:
            metadatos = json.load(f)
        
        # Cargar modelo
        modelo = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Sistema cargado correctamente")
        print(f"📊 Total de fragmentos: {len(metadatos)}")
        print(f"📁 Archivos procesados: {len(set(m['archivo'] for m in metadatos))}")
        
        return index, embeddings, contenido, metadatos, modelo
    
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontró el archivo {e.filename}")
        print("Asegúrate de haber ejecutado primero 'procesar_multiples_pdfs.py'")
        return None, None, None, None, None

def buscar_fragmentos_relevantes(pregunta, index, embeddings, metadatos, modelo, top_k=3):
    """Busca fragmentos relevantes para una pregunta"""
    pregunta_embedding = modelo.encode([pregunta])
    _, indices = index.search(pregunta_embedding, top_k)
    
    resultados = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadatos):
            resultados.append({
                'indice': idx,
                'archivo': metadatos[idx]['archivo'],
                'fragmento_numero': metadatos[idx]['fragmento_numero'],
                'similaridad': float(1 / (1 + _[0][i]))  # Convertir distancia a similaridad
            })
    
    return resultados

def mostrar_resultados(resultados, contenido):
    """Muestra los resultados de búsqueda"""
    if not resultados:
        print("❌ No se encontraron resultados relevantes")
        return
    
    print(f"\n🔍 Encontrados {len(resultados)} resultados:")
    
    for i, resultado in enumerate(resultados, 1):
        print(f"\n--- Resultado {i} ---")
        print(f"📄 Archivo: {resultado['archivo']}")
        print(f"📝 Fragmento: {resultado['fragmento_numero']}")
        print(f"📊 Similitud: {resultado['similaridad']:.2%}")
        
        # Extraer el fragmento del contenido
        inicio_marker = f"=== FRAGMENTO {resultado['indice']+1} ==="
        fin_marker = f"=== FRAGMENTO {resultado['indice']+2} ===" if resultado['indice']+2 <= len(contenido.split("===")) else ""
        
        lines = contenido.split('\n')
        fragmento_lines = []
        capturando = False
        
        for line in lines:
            if inicio_marker in line:
                capturando = True
                continue
            elif fin_marker in line:
                break
            elif capturando and line.strip():
                fragmento_lines.append(line)
        
        fragmento_texto = '\n'.join(fragmento_lines)
        print(f"📖 Contenido: {fragmento_texto[:300]}{'...' if len(fragmento_texto) > 300 else ''}")

def consulta_interactiva():
    """Interfaz interactiva para consultas"""
    print("🤖 Sistema de Búsqueda Semántica de PDFs")
    print("=" * 50)
    
    # Cargar sistema
    index, embeddings, contenido, metadatos, modelo = cargar_sistema()
    
    if not index:
        return
    
    print("\n💡 Ejemplos de preguntas que puedes hacer:")
    print("  - ¿Cuál es el impacto de las TIC en la agricultura?")
    print("  - ¿Qué dice sobre innovación en el sector agrícola?")
    print("  - ¿Cómo afectan las tecnologías a los agricultores?")
    print("  - ¿Qué metodología se usó en la investigación?")
    print("\n" + "=" * 50)
    
    while True:
        try:
            pregunta = input("\n❓ Ingresa tu pregunta (o 'salir' para terminar): ").strip()
            
            if pregunta.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta luego!")
                break
            
            if not pregunta:
                print("⚠️  Por favor ingresa una pregunta")
                continue
            
            print(f"\n🔍 Buscando: '{pregunta}'")
            
            # Realizar búsqueda
            resultados = buscar_fragmentos_relevantes(pregunta, index, embeddings, metadatos, modelo, top_k=3)
            
            # Mostrar resultados
            mostrar_resultados(resultados, contenido)
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    consulta_interactiva() 