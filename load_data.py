import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validate and check your MongoDB data"""
    
    def __init__(self):
        # Your MongoDB connection details
        username = "prithevikrishnamurali"
        password = "Db@12345678"  # URL encoded version: Db%4012345678
        cluster = "unidata.sggol9r.mongodb.net"
        database_name = "University_data"
        
        # Construct MongoDB URI
        self.mongodb_url = f"mongodb+srv://{username}:{quote_plus(password)}@{cluster}/"
        self.database_name = database_name
        self.client = None
        self.db = None
        
        # Province collections based on your actual structure
        self.provinces = [
            'British columbia',
            'New Brunswick', 
            'Newfoundland',
            'Nova scotia',
            'Ontario',
            'Prince Edward island',
            'Quebec',
            'Saskatchewan',
            'alberta',
            'manitoba'
        ]
    
    async def connect(self):
        """Connect to MongoDB Atlas"""
        try:
            self.client = AsyncIOMotorClient(self.mongodb_url)
            self.db = self.client[self.database_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"✅ Successfully connected to MongoDB Atlas")
            logger.info(f"📊 Database: {self.database_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            return False
    
    async def check_collections(self):
        """Check what collections exist in your database"""
        try:
            collections = await self.db.list_collection_names()
            logger.info(f"📁 Found {len(collections)} collections:")
            
            for collection in collections:
                count = await self.db[collection].count_documents({})
                logger.info(f"   - {collection}: {count} documents")
            
            return collections
            
        except Exception as e:
            logger.error(f"❌ Error checking collections: {e}")
            return []
    
    async def sample_data(self, collection_name, limit=3):
        """Sample some data from a collection"""
        try:
            logger.info(f"\n📋 Sample data from '{collection_name}':")
            cursor = self.db[collection_name].find({}).limit(limit)
            documents = await cursor.to_list(length=limit)
            
            for i, doc in enumerate(documents, 1):
                logger.info(f"   Sample {i}:")
                # Remove _id for cleaner output
                if '_id' in doc:
                    del doc['_id']
                
                # Print first few fields
                for key, value in list(doc.items())[:5]:
                    if isinstance(value, list) and len(value) > 3:
                        logger.info(f"     {key}: {value[:3]}... ({len(value)} total)")
                    else:
                        logger.info(f"     {key}: {value}")
                logger.info("")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error sampling data from {collection_name}: {e}")
            return []
    
    async def validate_data_structure(self):
        """Validate that your data has the expected structure"""
        logger.info("\n🔍 Validating data structure...")
        
        issues = []
        total_universities = 0
        
        for province in self.provinces:
            try:
                collection = self.db[province]
                count = await collection.count_documents({})
                total_universities += count
                
                if count == 0:
                    issues.append(f"⚠️  Collection '{province}' is empty")
                    continue
                
                # Sample one document to check structure
                sample = await collection.find_one({})
                if sample:
                    required_fields = ['name']  # Minimum required field
                    missing_fields = [field for field in required_fields if field not in sample]
                    
                    if missing_fields:
                        issues.append(f"⚠️  Collection '{province}' missing fields: {missing_fields}")
                    else:
                        logger.info(f"✅ Collection '{province}': {count} universities")
                
            except Exception as e:
                issues.append(f"❌ Error accessing collection '{province}': {e}")
        
        logger.info(f"\n📈 Total universities across all provinces: {total_universities}")
        
        if issues:
            logger.info("\n⚠️  Issues found:")
            for issue in issues:
                logger.info(f"   {issue}")
        else:
            logger.info("\n✅ All data validation checks passed!")
        
        return len(issues) == 0, total_universities
    
    async def create_indexes(self):
        """Create indexes for better performance"""
        logger.info("\n🔧 Creating indexes...")
        
        indexes_created = 0
        for province in self.provinces:
            try:
                collection = self.db[province]
                
                # Create indexes on commonly searched fields
                await collection.create_index([("name", 1)])
                if await collection.find_one({"programs": {"$exists": True}}):
                    await collection.create_index([("programs", 1)])
                if await collection.find_one({"location": {"$exists": True}}):
                    await collection.create_index([("location", 1)])
                
                indexes_created += 1
                logger.info(f"✅ Created indexes for '{province}'")
                
            except Exception as e:
                logger.warning(f"⚠️  Could not create indexes for '{province}': {e}")
        
        # Create session indexes
        try:
            await self.db.sessions.create_index([("session_id", 1)])
            await self.db.sessions.create_index([("created_at", 1)], expireAfterSeconds=3600*24)
            logger.info("✅ Created session management indexes")
        except Exception as e:
            logger.warning(f"⚠️  Could not create session indexes: {e}")
        
        logger.info(f"📊 Successfully created indexes for {indexes_created}/{len(self.provinces)} collections")
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 Database connection closed")

async def main():
    """Main function to validate your data"""
    validator = DataValidator()
    
    # Connect to database
    if not await validator.connect():
        return
    
    try:
        # Check collections
        collections = await validator.check_collections()
        
        # Sample data from a few collections
        if collections:
            # Try to sample from a few provinces
            sample_provinces = ['Ontario', 'British_columbia', 'Quebec']
            for province in sample_provinces:
                if province in collections:
                    await validator.sample_data(province)
                    break
        
        # Validate data structure
        is_valid, total_count = await validator.validate_data_structure()
        
        if is_valid and total_count > 0:
            logger.info("\n🎉 Your database is ready for the chatbot!")
            
            # Create indexes for better performance
            await validator.create_indexes()
            
        else:
            logger.info("\n❌ Please fix the data issues before running the chatbot")
    
    finally:
        await validator.close()

if __name__ == "__main__":
    print("🚀 Starting database validation...")
    print("=" * 50)
    asyncio.run(main())
    print("=" * 50)
    print("✨ Validation complete!")