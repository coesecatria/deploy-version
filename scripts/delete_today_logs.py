import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime

# Configuration from .env would be better, but for a one-off script:
MONGO_URI = "mongodb+srv://attend-ai:attend-ai@cluster0.3wsuy68.mongodb.net/"
DB_NAME = "attendance_db"
TODAY = "2026-03-10"

async def delete_today_logs():
    client = AsyncIOMotorClient(MONGO_URI, tlsAllowInvalidCertificates=True)
    db = client[DB_NAME]
    
    print(f"Connecting to {DB_NAME}...")
    
    # Count before deletion
    count_before = await db.attendance.count_documents({"date": TODAY})
    print(f"Found {count_before} attendance logs for {TODAY}.")
    
    if count_before > 0:
        result = await db.attendance.delete_many({"date": TODAY})
        print(f"Deleted {result.deleted_count} logs.")
    else:
        print("No logs to delete.")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(delete_today_logs())
