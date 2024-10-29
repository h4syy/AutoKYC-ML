import os
from dotenv import load_dotenv
import aiomysql

   # Load environment variables from .env file
load_dotenv()

   # Retrieve database configurations from environment variables
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT"))  # Ensure this is an integer
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")

   # Pool variable
db_pool = None

   # Initialize connection pool
async def init_db_pool():
       global db_pool
       db_pool = await aiomysql.create_pool(
           host=MYSQL_HOST,
           port=MYSQL_PORT,
           user=MYSQL_USER,
           password=MYSQL_PASSWORD,
           db=MYSQL_DB,
           minsize=1, 
           maxsize=10
       )

   # Cleanup connection pool
async def close_db_pool():
    global db_pool
    if db_pool:
        db_pool.close()
        await db_pool.wait_closed()