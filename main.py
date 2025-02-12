from datetime import datetime, timezone
import os
from fastapi import FastAPI, HTTPException, Query, Request
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch

# Load environment variables
load_dotenv()

# Fetch API key and Mongo URI from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Initialize FastAPI
app = FastAPI()

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["servicio"]
collection = db["systems"]
embeddings_collection = db["embeddings"]

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

@app.get("/")
def home():
    return {
        "message": "Welcome to the AI-powered server!",
        "mongo_status": "Connected" if client else "Not connected"
    }

@app.post("/system-created")
async def system_created(request: Request):
    try:
        # Read the incoming JSON data
        system_data = await request.json()
        
        # Log the received data
        print("Received system data:", json.dumps(system_data, indent=2))

        # Simulate processing (e.g., saving to MongoDB)
        # db.systems.insert_one(system_data)  # Uncomment if using MongoDB

        return {"message": "Python received the system data successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate-embeddings/{system_id}")
def generate_and_store_embeddings(system_id: str):
    try:
        obj_id = ObjectId(system_id)
        category = "system"
    except:
        raise HTTPException(status_code=400, detail="Formato de ObjectId inválido.")

    system = collection.find_one({"_id": obj_id})
    if not system:
        raise HTTPException(status_code=404, detail="Sistema no encontrado.")

    # Asegurarnos de que los datos sean descriptivos
    system_name = system.get("system_name", "Desconocido")
    system_code = system.get("system_code", "No especificado")
    acquisition_date = system.get("acquisition_date", "Desconocida")

    if isinstance(acquisition_date, datetime):
        acquisition_date = acquisition_date.isoformat()

    text = f"{system_name}, código {system_code}, adquirido en {acquisition_date}."

    embedding = embeddings.embed_query(text)  # Generar embedding con OpenAI

    embeddings_collection.insert_one({
        "document_id": obj_id,
        "text": text,
        "category": category,
        "embedding": embedding,
        "created_at": system.get("created_at", datetime.now(timezone.utc)).isoformat(),
        "updated_at": system.get("updated_at", datetime.now(timezone.utc)).isoformat()
    })

@app.get("/get-embeddings/{system_id}")
def get_embeddings(system_id: str):
    try:
        obj_id = ObjectId(system_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    document = embeddings_collection.find_one({"document_id": obj_id})

    if document:
        document["_id"] = str(document["_id"])
        document["document_id"] = str(document["document_id"]) 
        return document
    else:
        raise HTTPException(status_code=404, detail="document not found")
    
@app.get("/search-2/")
def vector_search(query: str = Query(..., description="Texto a buscar en la base de datos")):
    """
    Realiza una búsqueda semántica en MongoDB usando `$vectorSearch` y siempre devuelve el resultado más similar.
    """
    try:
        query_embedding = embeddings.embed_query(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 200,
                    "limit": 5, 
                    "metric": "cosine"
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "document_id": 1,
                    "category": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
            {
                "$sort": {"score": -1} 
            }
        ]

        results = list(collection.aggregate(pipeline))

        if results:
            return {
                "query": query,
                "similar_systems": results
            }
        
        return {
            "query": query,
            "message": "No se encontraron coincidencias exactas, pero aquí está lo más cercano:",
            "similar_systems": [results[0]] if results else []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
  
# Initialize OpenAI LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY, max_tokens=50)  # Limit tokens to save costs

@app.get("/search")
def vector_search(query: str = Query(..., description="Texto a buscar en la base de datos")):
    """
    Busca documentos similares en la base de datos usando embeddings.
    """
    try:
        query_embedding = obtener_embedding(query)

        results = buscar_en_mongodb(query_embedding)
        
        if results:
            result = generar_respuesta(query, results)
            return {
                "query": query,
                "similar_systems": result
            }

        return {
            "query": query,
            "message": "No se encontraron coincidencias exactas, pero aquí está lo más cercano:",
            "similar_systems": [results[0]] if results else []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def obtener_embedding(consulta):
    return embeddings.embed_query(consulta)

def buscar_en_mongodb(query_embedding):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": 5,
                "metric": "cosine"
            }
        },
        {
            "$project": {
                "_id": 0, 
                "document_id": 1,
                "text": 1, 
                "category": 1
            }
        }
    ]

    return list(embeddings_collection.aggregate(pipeline))

def generar_respuesta(consulta, results):
    resultados_texto = "\n".join(
        [f"- idenfiticador: {r['document_id']}, Categoria: {r['category']}, Texto: {r['text']}" for r in results]
    )
    prompt = f"Un usuario preguntó: \"{consulta}\". Basado en los siguientes datos encontrados, responde de manera útil:\n{resultados_texto}"
    
    return llm.predict(prompt)