from fastapi import FastAPI
from routers import document_detection_front, document_detection_back, face_comparision, liveness
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from database import dbconfig

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await dbconfig.init_db_pool()
    yield
    await dbconfig.close_db_pool()

app = FastAPI(lifespan=lifespan)

# Environment variables
base_url = os.getenv("BASE_URL", "/api/v1")  # Fallback to "/api/v1"
host = os.getenv("HOST", "0.0.0.0")         # Fallback to "0.0.0.0"
port = int(os.getenv("PORT", 8000))         # Fallback to 8000

# Include routers
app.include_router(document_detection_front.router, prefix=base_url)
app.include_router(document_detection_back.router, prefix=base_url)
app.include_router(face_comparision.router, prefix=base_url)
app.include_router(liveness.router, prefix=base_url)

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=host, port=port)
