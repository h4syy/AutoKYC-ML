import logging
import asyncio
import aiomysql
from database import dbconfig

class MySQLHandler(logging.Handler):
    def __init__(self, db_config):
        super().__init__()
        self.db_config = db_config

    def emit(self, record):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._async_emit(record))
        else:
            loop.run_until_complete(self._async_emit(record))

    async def _async_emit(self, record):
        log_entry = self.format(record)
        try:
            async with dbconfig.db_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "INSERT INTO logs (level, message) VALUES (%s, %s)",
                        (record.levelname, log_entry)
                    )
                    await conn.commit()
        except Exception as e:
            print(f"Failed to log to database: {e}")

# Logger setup
logger = logging.getLogger("db_logger")
logger.setLevel(logging.INFO)
logger_handler = MySQLHandler(dbconfig.db_config)
logger_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(logger_handler)