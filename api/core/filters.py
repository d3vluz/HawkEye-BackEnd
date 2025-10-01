from PIL import Image, ImageFilter, ImageStat
import numpy as np
from typing import Dict, Any, Tuple

def apply_threshold(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Aplica limiarização (threshold) e retorna análise.
    
    Returns:
        Tupla com (imagem_processada, dados_análise)
    """
    gray_image = image.convert("L")

    histogram = gray_image.histogram()
    total_pixels = sum(histogram)
    
    threshold_value = 128
    sum_total = sum(i * histogram[i] for i in range(256))
    threshold_value = int(sum_total / total_pixels)
    
    threshold_image = gray_image.point(lambda x: 255 if x > threshold_value else 0)
    
    white_pixels = sum(histogram[i] for i in range(threshold_value + 1, 256))
    black_pixels = total_pixels - white_pixels
    
    analysis_data = {
        "threshold_value": threshold_value,
        "white_pixels_percentage": round((white_pixels / total_pixels) * 100, 2),
        "black_pixels_percentage": round((black_pixels / total_pixels) * 100, 2),
        "total_pixels": total_pixels,
        "description": f"Limiarização aplicada com valor de corte: {threshold_value}"
    }
    
    return threshold_image, analysis_data

def apply_edge_detection(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Aplica detecção de bordas e retorna análise.
    
    Returns:
        Tupla com (imagem_processada, dados_análise)
    """
    gray_image = image.convert("L")
    
    edges = gray_image.filter(ImageFilter.FIND_EDGES)

    stat = ImageStat.Stat(edges)
    mean_intensity = stat.mean[0]
    
    edges_array = np.array(edges)
    edge_pixels = np.count_nonzero(edges_array > 30)
    total_pixels = edges_array.size
    
    analysis_data = {
        "edge_density": round((edge_pixels / total_pixels) * 100, 2),
        "mean_edge_intensity": round(mean_intensity, 2),
        "edge_pixel_count": int(edge_pixels),
        "complexity_score": round(mean_intensity / 2.55, 2),
        "description": "Detecção de bordas usando filtro FIND_EDGES"
    }
    
    return edges, analysis_data

def apply_noise_reduction(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Aplica redução de ruído e retorna análise.
    
    Returns:
        Tupla com (imagem_processada, dados_análise)
    """
    original_stat = ImageStat.Stat(image)
    original_stddev = original_stat.stddev
    
    denoised = image.filter(ImageFilter.MedianFilter(size=3))
    
    denoised_stat = ImageStat.Stat(denoised)
    denoised_stddev = denoised_stat.stddev
    
    if len(original_stddev) == 3:
        noise_reduction = [
            round(((o - d) / o * 100), 2) if o > 0 else 0
            for o, d in zip(original_stddev, denoised_stddev)
        ]
        avg_noise_reduction = round(sum(noise_reduction) / 3, 2)
    else:
        avg_noise_reduction = round(((original_stddev[0] - denoised_stddev[0]) / original_stddev[0] * 100), 2)
        noise_reduction = [avg_noise_reduction]
    
    analysis_data = {
        "noise_reduction_percentage": avg_noise_reduction,
        "original_noise_level": round(sum(original_stddev) / len(original_stddev), 2),
        "processed_noise_level": round(sum(denoised_stddev) / len(denoised_stddev), 2),
        "filter_applied": "Median Filter (3x3)",
        "description": "Redução de ruído usando filtro mediano"
    }
    
    return denoised, analysis_data

def apply_histogram_analysis(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Analisa o histograma da imagem e retorna equalização.
    
    Returns:
        Tupla com (imagem_equalizada, dados_análise)
    """
    from PIL import ImageOps
    
    stat = ImageStat.Stat(image)
    equalized = ImageOps.equalize(image)
    equalized_stat = ImageStat.Stat(equalized)
    
    analysis_data = {
        "original_mean": [round(m, 2) for m in stat.mean],
        "original_median": [round(m, 2) for m in stat.median],
        "original_stddev": [round(s, 2) for s in stat.stddev],
        "equalized_mean": [round(m, 2) for m in equalized_stat.mean],
        "contrast_improvement": round(
            (sum(equalized_stat.stddev) - sum(stat.stddev)) / sum(stat.stddev) * 100, 2
        ),
        "description": "Análise de histograma com equalização"
    }
    
    return equalized, analysis_data

def apply_brightness_contrast_analysis(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Analisa brilho e contraste da imagem.
    
    Returns:
        Tupla com (imagem_ajustada, dados_análise)
    """
    from PIL import ImageEnhance
    
    stat = ImageStat.Stat(image)
    
    avg_brightness = sum(stat.mean) / len(stat.mean)
    brightness_percentage = (avg_brightness / 255) * 100
    
    avg_contrast = sum(stat.stddev) / len(stat.stddev)
    contrast_percentage = (avg_contrast / 128) * 100

    enhanced = image.copy()
    adjustments_made = []
    
    if brightness_percentage < 40:
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(1.2)
        adjustments_made.append("brightness +20%")
    elif brightness_percentage > 70:
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(0.9)
        adjustments_made.append("brightness -10%")
    
    if contrast_percentage < 30:
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.3)
        adjustments_made.append("contrast +30%")
    
    analysis_data = {
        "brightness_level": round(brightness_percentage, 2),
        "contrast_level": round(contrast_percentage, 2),
        "brightness_classification": (
            "Muito escura" if brightness_percentage < 30 else
            "Escura" if brightness_percentage < 45 else
            "Normal" if brightness_percentage < 65 else
            "Clara" if brightness_percentage < 80 else
            "Muito clara"
        ),
        "contrast_classification": (
            "Baixo" if contrast_percentage < 25 else
            "Médio" if contrast_percentage < 50 else
            "Alto"
        ),
        "adjustments_applied": adjustments_made if adjustments_made else ["Nenhum ajuste necessário"],
        "description": "Análise de brilho e contraste"
    }
    
    return enhanced, analysis_data