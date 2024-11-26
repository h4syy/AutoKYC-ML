import aiomysql
import os

db_config = {
       "host": os.getenv("MYSQL_HOST"),
       "port": int(os.getenv("MYSQL_PORT", 3306)),
       "user": os.getenv("MYSQL_USER"),
       "password": os.getenv("MYSQL_PASSWORD"),
       "db": os.getenv("MYSQL_DB"),
   }

async def init_db_pool():
       global db_pool
       db_pool = await aiomysql.create_pool(
           host=db_config["host"],
           port=db_config["port"],
           user=db_config["user"],
           password=db_config["password"],
           db=db_config["db"],
           autocommit=True,
       )