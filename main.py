from datetime import datetime, timezone
import os

from openai import BaseModel
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
    
class EmbeddingRequest(BaseModel):
    id: str
    category: str
    summary: str
    
@app.post("/generate-embeddings")
def generate_and_store_embeddings(request: EmbeddingRequest):
    try:
        obj_id = ObjectId(request.id)
    except:
        raise HTTPException(status_code=400, detail="Formato de ObjectId inválido.")
    
    if not request.summary:
        raise HTTPException(status_code=400, detail="El campo 'summary' es obligatorio.")
    
    now = datetime.now(timezone.utc).isoformat()
    
    embedding = embeddings.embed_query(request.summary)
    
    document = {
        "document_id": obj_id,
        "text": request.summary,
        "category": request.category,
        "embedding": embedding,
        "created_at": now,
        "updated_at": now
    }

    embeddings_collection.insert_one(document)

    return {
        "document_id": str(obj_id),
        "text": request.summary,
        "category": request.category,
        "embedding": embedding,
        "created_at": now,
        "updated_at": now
    }

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
    
# Initialize OpenAI LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY, max_tokens=100)  # Limit tokens to save costs

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
        [f"Categoria: {r['category']}, Texto: {r['text']}" for r in results]
    )
    prompt = f"Un usuario preguntó: \"{consulta}\". Basado en los siguientes datos encontrados, responde de manera precisa, corta y clara:\n{resultados_texto}, sino encuentras relación no fuerces la respuesta"
    
    return llm.predict(prompt)

def get_system(_id: ObjectId):
    document = db["system"].find_one({"_id": _id})
    
    if not document:
        raise HTTPException(status_code=404, detail="Sistema no encontrado.")

    # detalle del sistema
    system_name = document.get("system_name")
    system_type = str(document.get("system_type", {}).get("$oid", "Sin tipo de sistema"))
    system_code = document.get("system_code", "No cuenta con código el sistema")

    acquisition_date = format_date(document.get("acquisition_date"))
    delivery_date = format_date(document.get("delivery_date"))
    created_at = format_date(document.get("created_at"))
    updated_at = format_date(document.get("updated_at"))

    # cliente
    customer = document.get("customer", {})
    customer_id = str(customer.get("customer_id", {}).get("$oid", "No especificado"))
    branch_id = str(customer.get("branch_id", {}).get("$oid", "No especificado"))
    department_id = str(customer.get("department_id", {}).get("$oid", "No especificado"))

    # Equipos
    equipments = document.get("equipments", [])
    equipment_details = []
    for eq in equipments:
        eq_name = eq.get("equipment_name", "Equipo desconocido")
        eq_serial = eq.get("serial_equipment", "No especificado")
        eq_model = eq.get("model", "No especificado")
        eq_price = eq.get("price", {}).get("$numberDouble", "0.0")
        eq_acquisition_date = format_date(eq.get("acquisition_date"))
        equipment_details.append(f"{eq_name} (Modelo: {eq_model}, Serie: {eq_serial}, Precio: ${eq_price}, Adquirido: {eq_acquisition_date})")

    equipment_text = " | ".join(equipment_details) if equipment_details else "Sin equipos asociados."

    # Construcción del texto para embeddings
    text = (
        f"Este sistema se llama {system_name} con código {system_code}. "
        f"Es del tipo {system_type} y fue adquirido el {acquisition_date}. "
        f"Se entregó el {delivery_date}. Pertenece al cliente con ID {customer_id}, "
        f"sucursal {branch_id} y departamento {department_id}. "
        f"Equipos asociados: {equipment_text}. "
        f"Documento creado el {created_at} y actualizado el {updated_at}."
    )

    return {
        "text": text,
        "created_at": created_at,
        "updated_at": updated_at
    }

def format_date(timestamp):
    """Convierte timestamps de MongoDB a formato ISO 8601."""
    if isinstance(timestamp, dict) and "$date" in timestamp:
        return datetime.fromtimestamp(timestamp["$date"]["$numberLong"] / 1000, tz=timezone.utc).isoformat()
    return "Sin fecha"