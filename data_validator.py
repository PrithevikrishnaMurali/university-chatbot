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
        # Get MongoDB connection details from environment variables
        self.mongodb_url = os.getenv("MONGODB_URL")
        self.database_name = os.getenv("DATABASE_NAME", "University_data")
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
        
        if not self.mongodb_url:
            logger.error("❌ MONGODB_URL not found in environment variables!")
            logger.error("Please create a .env file with your MongoDB connection string.")
            raise ValueError("MongoDB URL is required")
    
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
            
            total_documents = 0
            for collection in collections:
                count = await self.db[collection].count_documents({})
                total_documents += count
                logger.info(f"   - {collection}: {count} documents")
            
            logger.info(f"📈 Total documents across all collections: {total_documents}")
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
                for key, value in list(doc.items())[:6]:
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
        province_counts = {}
        
        for province in self.provinces:
            try:
                collection = self.db[province]
                count = await collection.count_documents({})
                province_counts[province] = count
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
                        issues.append(f"⚠️  Collection '{province}' missing required fields: {missing_fields}")
                    else:
                        logger.info(f"✅ Collection '{province}': {count} universities")
                        
                    # Check for common fields
                    common_fields = ['city', 'province', 'country', 'number_of_students']
                    present_fields = [field for field in common_fields if field in sample]
                    if present_fields:
                        logger.info(f"   📝 Available fields: {', '.join(present_fields)}")
                
            except Exception as e:
                issues.append(f"❌ Error accessing collection '{province}': {e}")
        
        logger.info(f"\n📊 University Distribution by Province:")
        logger.info("=" * 40)
        for province, count in sorted(province_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / total_universities * 100) if total_universities > 0 else 0
                logger.info(f"  {province:20} {count:3d} ({percentage:5.1f}%)")
        
        logger.info(f"\n📈 Total universities across all provinces: {total_universities}")
        
        if issues:
            logger.info("\n⚠️  Issues found:")
            for issue in issues:
                logger.info(f"   {issue}")
            
            # Check if collections exist with slightly different names
            logger.info("\n🔍 Checking for similar collection names...")
            all_collections = await self.db.list_collection_names()
            for province in self.provinces:
                if province not in all_collections:
                    # Look for similar names
                    similar = [c for c in all_collections if province.lower().replace(' ', '') in c.lower().replace(' ', '').replace('_', '')]
                    if similar:
                        logger.info(f"   💡 '{province}' not found, but found similar: {similar}")
        else:
            logger.info("\n✅ All data validation checks passed!")
        
        return len(issues) == 0, total_universities
    
    async def create_indexes(self):
        """Create indexes for better performance"""
        logger.info("\n🔧 Creating indexes for better performance...")
        
        indexes_created = 0
        for province in self.provinces:
            try:
                collection = self.db[province]
                
                # Check if collection has documents
                count = await collection.count_documents({})
                if count == 0:
                    continue
                
                # Create indexes on commonly searched fields
                await collection.create_index([("name", 1)])
                
                # Check if these fields exist before creating indexes
                sample = await collection.find_one({})
                if sample:
                    if 'programs' in sample:
                        await collection.create_index([("programs", 1)])
                    if 'city' in sample:
                        await collection.create_index([("city", 1)])
                    if 'province' in sample:
                        await collection.create_index([("province", 1)])
                    if 'number_of_students' in sample:
                        await collection.create_index([("number_of_students", 1)])
                
                indexes_created += 1
                logger.info(f"✅ Created indexes for '{province}' ({count} universities)")
                
            except Exception as e:
                logger.warning(f"⚠️  Could not create indexes for '{province}': {e}")
        
        # Create session indexes for the chatbot
        try:
            await self.db.sessions.create_index([("session_id", 1)])
            await self.db.sessions.create_index([("created_at", 1)], expireAfterSeconds=3600*24)  # 24 hours
            logger.info("✅ Created session management indexes")
        except Exception as e:
            logger.warning(f"⚠️  Could not create session indexes: {e}")
        
        logger.info(f"📊 Successfully created indexes for {indexes_created} collections")
    
    async def test_chatbot_compatibility(self):
        """Test compatibility with the chatbot requirements"""
        logger.info("\n🤖 Testing chatbot compatibility...")
        
        # Test if we can load universities like the chatbot does
        total_loaded = 0
        compatibility_issues = []
        
        for province in self.provinces:
            try:
                cursor = self.db[province].find({})
                universities = await cursor.to_list(length=None)
                
                if universities:
                    total_loaded += len(universities)
                    
                    # Test serialization (like the chatbot does)
                    for uni in universities[:1]:  # Test one university per province
                        # Remove ObjectId to test serialization
                        if '_id' in uni:
                            del uni['_id']
                        
                        # Test that required fields are accessible
                        name = uni.get('name', 'Unknown')
                        if name == 'Unknown':
                            compatibility_issues.append(f"University in '{province}' missing 'name' field")
                    
                    logger.info(f"✅ '{province}': {len(universities)} universities loaded successfully")
                else:
                    logger.info(f"⚠️  '{province}': No universities found")
                    
            except Exception as e:
                compatibility_issues.append(f"Error loading '{province}': {e}")
                logger.error(f"❌ Error loading '{province}': {e}")
        
        logger.info(f"\n📊 Chatbot Compatibility Results:")
        logger.info(f"   Total universities loaded: {total_loaded}")
        
        if compatibility_issues:
            logger.info(f"   Issues found: {len(compatibility_issues)}")
            for issue in compatibility_issues:
                logger.info(f"     - {issue}")
            return False
        else:
            logger.info("   ✅ All compatibility checks passed!")
            return True
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 Database connection closed")

async def main():
    """Main function to validate your data"""
    validator = DataValidator()
    
    print("🚀 Starting database validation...")
    print("=" * 50)
    
    # Connect to database
    if not await validator.connect():
        return
    
    try:
        # Check collections
        collections = await validator.check_collections()
        
        # Sample data from a few collections
        if collections:
            # Try to sample from a few provinces that have data
            sample_provinces = ['Ontario', 'Quebec', 'British columbia', 'alberta']
            for province in sample_provinces:
                if province in collections:
                    await validator.sample_data(province, limit=2)
                    break
        
        # Validate data structure
        is_valid, total_count = await validator.validate_data_structure()
        
        # Test chatbot compatibility
        is_compatible = await validator.test_chatbot_compatibility()
        
        if is_valid and is_compatible and total_count > 0:
            logger.info("\n🎉 Your database is ready for the chatbot!")
            
            # Create indexes for better performance
            await validator.create_indexes()
            
            logger.info("\n🚀 Next steps:")
            logger.info("   1. Run: python debug_backend.py")
            logger.info("   2. Run: python main.py")
            logger.info("   3. Open: chatbot_ui.html")
            
        else:
            logger.info("\n❌ Please fix the issues before running the chatbot")
            if total_count == 0:
                logger.info("   🔄 Make sure your university data is loaded in MongoDB")
            if not is_valid:
                logger.info("   🔧 Fix the data structure issues listed above")
            if not is_compatible:
                logger.info("   🤖 Fix the chatbot compatibility issues")
    
    finally:
        await validator.close()

if __name__ == "__main__":
    print("🎓 University Chatbot - Database Validator")
    print("==========================================")
    asyncio.run(main())
    print("=" * 50)
    print("✨ Validation complete!")