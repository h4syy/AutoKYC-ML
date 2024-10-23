# main.py (or app.py)
from fastapi import FastAPI
from routes import inference_router, face_compare_routes  # Import both routers

app = FastAPI()

# Include the routes for inference
app.include_router(inference_router.router, prefix="/inference")

# Include the routes for face comparison
app.include_router(face_compare_routes.router, prefix="/face")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
