from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any

class AnalysisFilter(str, Enum):
    """Enum com os filtros de análise disponíveis."""
    THRESHOLD = "threshold"
    EDGE_DETECTION = "edge_detection"
    NOISE_REDUCTION = "noise_reduction"
    HISTOGRAM = "histogram"
    BRIGHTNESS_CONTRAST = "brightness_contrast"

class FilterAnalysisResult(BaseModel):
    """Resultado da análise de um filtro específico."""
    filter_name: str
    analysis_data: Dict[str, Any]
    processed_image_data: str

class ImageAnalysisRequest(BaseModel):
    """Schema de requisição para análise de imagem."""
    image_data: str
    filters: List[AnalysisFilter]

class ImageAnalysisResponse(BaseModel):
    """Schema de resposta com todas as análises."""
    original_image_data: str
    results: List[FilterAnalysisResult]