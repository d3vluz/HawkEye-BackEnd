from PIL import Image
from typing import Callable, Tuple, Dict, Any
from . import analysis_filters
from models.schemas import AnalysisFilter

ANALYSIS_FILTER_MAP: Dict[AnalysisFilter, Callable[[Image.Image], Tuple[Image.Image, Dict[str, Any]]]] = {
    AnalysisFilter.THRESHOLD: analysis_filters.apply_threshold,
    AnalysisFilter.EDGE_DETECTION: analysis_filters.apply_edge_detection,
    AnalysisFilter.NOISE_REDUCTION: analysis_filters.apply_noise_reduction,
    AnalysisFilter.HISTOGRAM: analysis_filters.apply_histogram_analysis,
    AnalysisFilter.BRIGHTNESS_CONTRAST: analysis_filters.apply_brightness_contrast_analysis,
}

def apply_analysis_filter(image: Image.Image, filter_name: AnalysisFilter) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Aplica um filtro de análise na imagem e retorna a imagem processada + dados da análise.

    Args:
        image: A imagem original (objeto PIL.Image).
        filter_name: O nome do filtro a ser aplicado (do Enum AnalysisFilter).

    Returns:
        Tupla contendo (imagem_processada, dados_análise).
        
    Raises:
        ValueError: Se o nome do filtro não for encontrado.
    """
    filter_function = ANALYSIS_FILTER_MAP.get(filter_name)
    
    if not filter_function:
        raise ValueError(f"Filtro de análise desconhecido: {filter_name}")
        
    return filter_function(image.copy())