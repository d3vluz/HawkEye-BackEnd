import io
import base64
from fastapi import APIRouter, HTTPException
from PIL import Image
import logging

from core.services import apply_analysis_filter
from models.schemas import (
    ImageAnalysisRequest,
    ImageAnalysisResponse,
    FilterAnalysisResult,
    AnalysisFilter
)

router = APIRouter()
logger = logging.getLogger(__name__)

def base64_to_image(base64_string: str) -> Image.Image:
    """
    Converte uma string base64 em objeto PIL Image.
    
    Args:
        base64_string: String base64 da imagem (com ou sem prefixo data:image)
    
    Returns:
        Objeto PIL.Image
    """
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Erro ao decodificar imagem base64: {str(e)}")


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Converte um objeto PIL Image em string base64.
    
    Args:
        image: Objeto PIL.Image
        format: Formato da imagem (PNG, JPEG, etc)
    
    Returns:
        String base64 com prefixo data:image
    """
    buffered = io.BytesIO()
    image_to_save = image

    if format.upper() == "JPEG" and image.mode in ("RGBA", "P", "L"):
        if image.mode == "L":
            image_to_save = image.convert("RGB")
        else:
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "RGBA":
                rgb_image.paste(image, mask=image.getchannel('A'))
            else:
                rgb_image.paste(image)
            image_to_save = rgb_image
    
    if format.upper() == "PNG" and image.mode == "L":
        image_to_save = image.convert("RGB")
    
    image_to_save.save(buffered, format=format, quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{img_str}"


@router.post(
    "/analyze-image/",
    response_model=ImageAnalysisResponse,
    summary="Analisa uma imagem com múltiplos filtros"
)
async def analyze_image(request: ImageAnalysisRequest):
    """
    Processa uma imagem com múltiplos filtros de análise.
    
    Cada filtro analisa a imagem original e retorna:
    - Dados específicos da análise
    - Imagem processada pelo filtro
    
    Args:
        request: Contém a imagem em base64 e lista de filtros a aplicar
    
    Returns:
        Resposta com a imagem original e resultados de cada filtro
    """
    if not request.filters:
        raise HTTPException(
            status_code=400,
            detail="Nenhum filtro foi especificado. Forneça ao menos um filtro."
        )
    
    if len(request.filters) > 10:
        raise HTTPException(
            status_code=400,
            detail="Máximo de 10 filtros por requisição."
        )

    try:
        original_image = base64_to_image(request.image_data)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Imagem base64 inválida: {str(e)}"
        )

    image_format = original_image.format or "PNG"
    
    results = []
    for filter_name in request.filters:
        try:
            processed_image, analysis_data = apply_analysis_filter(
                original_image, 
                filter_name
            )
            
            processed_base64 = image_to_base64(processed_image, format=image_format)
            
            results.append(
                FilterAnalysisResult(
                    filter_name=filter_name.value,
                    analysis_data=analysis_data,
                    processed_image_data=processed_base64
                )
            )
            
            logger.info(f"Filtro '{filter_name.value}' aplicado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao aplicar filtro '{filter_name.value}': {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao processar filtro '{filter_name.value}': {str(e)}"
            )
    
    return ImageAnalysisResponse(
        original_image_data=request.image_data,
        results=results
    )