from starlette.middleware.base import BaseHTTPMiddleware
from aiomysql import create_pool
from configs.db_config import db_config

class DBMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.db_pool = None

    async def dispatch(self, request, call_next):
        # Create the pool if not already created
        if not self.db_pool:
            self.db_pool = await create_pool(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                db=db_config['database']
            )
        
        # Attach the pool to the request's state
        request.state.db_pool = self.db_pool
        
        # Process the request
        response = await call_next(request)
        
        return response
