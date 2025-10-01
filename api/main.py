from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from routes import images 

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Image Analysis API",
    description="API para análise de imagens com múltiplos filtros especializados.",
    version="3.0.0"
)

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(images.router, tags=["Image Analysis"])

@app.get("/", summary="Verifica a saúde da API")
async def root():
    return {
        "message": "Image Analysis API está funcionando!",
        "version": "2.0.0",
        "available_filters": [
            "threshold",
            "edge_detection",
            "noise_reduction",
            "histogram",
            "brightness_contrast"
        ]
    }