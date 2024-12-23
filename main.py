from fastapi import FastAPI
from routers import document_detection_front, document_detection_back, face_comparision, liveness, backup
from database import dbconfig
from contextlib import asynccontextmanager
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    await dbconfig.init_db_pool()
    yield  
    await dbconfig.close_db_pool()

app = FastAPI(lifespan=lifespan)

base_url = os.getenv("BASE_URL") 

app.include_router(document_detection_front.router, prefix=base_url)
app.include_router(document_detection_back.router, prefix=base_url)
app.include_router(face_comparision.router, prefix=base_url)
app.include_router(liveness.router, prefix=base_url)

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")  
    port = int(os.getenv("PORT", 8000))  
    uvicorn.run(app, host=host, port=port)
