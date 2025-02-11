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


""" @app.get("/movie/{movie_id}")
def get_movie_by_id(movie_id: str):
    try:
        obj_id = ObjectId(movie_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    movie = collection.find_one({"_id": obj_id})

    if movie:
        movie["_id"] = str(movie["_id"])  # Convert ObjectId to string
        return movie
    else:
        raise HTTPException(status_code=404, detail="Movie not found") """


""" @app.get("/search/")
def vector_search(query: str = Query(..., description="Text to search for similar movies")):

    # Find similar movies using MongoDBAtlasVectorSearch
    results = vector_store.similarity_search(query, k=3)  # Get top 3 most similar movies

    if not results:
        return {"message": "No similar movies found."}

    # Step 2: Generate a human-like response
    movie_titles = [movie.metadata["title"] for movie in results]
    context = ", ".join(movie_titles)
    prompt = f"The user is looking for movies similar to '{query}'. The most relevant movies are: {context}. Explain briefly why they are related."
    
    ai_response = llm(prompt)  # OpenAI LLM API call

    # Convert ObjectId to string
    formatted_results = [
        {"_id": str(movie.metadata["_id"]), "title": movie.metadata["title"], "plot": movie.page_content}
        for movie in results
    ]

    return {
        "query": query,
        "related_movies": formatted_results,
        "response": ai_response 
    } """

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
    """
    Genera y almacena embeddings para un sistema si aún no existen en la base de datos.
    """
    try:
        obj_id = ObjectId(system_id)
        category = "system"
    except:
        raise HTTPException(status_code=400, detail="Formato de ObjectId inválido.")

    existing_embedding = embeddings_collection.find_one({"document_id": obj_id})
    if existing_embedding:
        return {"message": "Embeddings ya existen para este documento.", "document_id": obj_id}

    system = collection.find_one({"_id": obj_id})
    if not system:
        raise HTTPException(status_code=404, detail="Sistema no encontrado.")

    # datos para el sistema
    system_name = system.get("system_name", "Desconocido")
    system_code = system.get("system_code", "No especificado")
    acquisition_date = system.get("acquisition_date", "Desconocida")

    if isinstance(acquisition_date, datetime):
        acquisition_date = acquisition_date.isoformat()

    text = f"Nombre: {system_name}, Código: {system_code}, Adquisición: {acquisition_date}."

    embedding = embeddings.embed_query(text)

    embeddings_collection.insert_one({
        "document_id": obj_id,
        "text": text,
        "category": category,
        "embedding": embedding,
        "created_at": system.get("created_at", datetime.now(timezone.utc)).isoformat(),
        "updated_at": system.get("updated_at", datetime.now(timezone.utc)).isoformat()
    })

    return {"message": "Embeddings generados y almacenados correctamente.", "document_id": system_id}

    
@app.get("/get-embeddings/{system_id}")
def get_embeddings(document_id: str):
    try:
        obj_id = ObjectId(document_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    document = embeddings_collection.find_one({"document_id": obj_id})

    if document:
        document["_id"] = str(document["_id"])
        document["document_id"] = str(document["document_id"]) 
        return document
    else:
        raise HTTPException(status_code=404, detail="document not found")
    
@app.get("/search/")
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