import json
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os

async def load_universities_from_json(json_file_path: str, mongodb_url: str = "mongodb+srv://prithevikrishnamurali:Db@12345678@unidata.sggol9r.mongodb.net/"):
    client = AsyncIOMotorClient(mongodb_url)
    db = client.university_chatbot
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            universities_data = json.load(file)
        
        await db.universities.delete_many({})
        
        if isinstance(universities_data, list):
            await db.universities.insert_many(universities_data)
            print(f"Successfully loaded {len(universities_data)} universities")
        else:
            print("Error: JSON file should contain a list of universities")
        
        await db.universities.create_index([("name", 1)])
        await db.universities.create_index([("location", 1)])
        await db.universities.create_index([("programs", 1)])
        await db.universities.create_index([("ranking", 1)])
        
        print("Indexes created successfully")
        
    except Exception as e:
        print(f"Error loading data: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    json_file = "your_universities.json"  # Update this path
    mongodb_url = "mongodb://localhost:27017"
    
    asyncio.run(load_universities_from_json(json_file, mongodb_url))