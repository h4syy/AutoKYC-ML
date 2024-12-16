from fastapi import FastAPI
from routers import document_detection_front, document_detection_back, face_comparision, liveness
from database import dbconfig
import os

app = FastAPI()

app.include_router(document_detection_front.router, prefix=os.getenv("BASE_URL"))
app.include_router(document_detection_back.router, prefix=os.getenv("BASE_URL"))
app.include_router(face_comparision.router, prefix=os.getenv("BASE_URL"))
app.include_router(liveness.router, prefix=os.getenv("BASE_URL"))

@app.on_event("startup")
async def on_startup():
    await dbconfig.init_db_pool()

@app.on_event("shutdown")
async def on_shutdown():
    await dbconfig.close_db_pool()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port= os.getenv("PORT"))