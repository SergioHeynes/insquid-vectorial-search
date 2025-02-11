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
db = client["sample_mflix"]
collection = db["embedded_movies"]

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize OpenAI LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY, max_tokens=50)  # Limit tokens to save costs

# Initialize MongoDB Atlas Vector Search
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index",  # Ensure this matches MongoDB index
    text_key="plot",  # Field storing the text
    embedding_key="plot_embedding"  # Field storing the vector embeddings
)


@app.get("/")
def home():
    return {
        "message": "Welcome to the AI-powered server!",
        "mongo_status": "Connected" if client else "Not connected"
    }


@app.get("/movie/{movie_id}")
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
        raise HTTPException(status_code=404, detail="Movie not found")


@app.get("/search/")
def vector_search(query: str = Query(..., description="Text to search for similar movies")):
    """
    Search for movies based on a text query using MongoDBAtlasVectorSearch.
    """

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