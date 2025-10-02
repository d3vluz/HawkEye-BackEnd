"""
@module main
Descrição: API FastAPI para processamento em lote de imagens. Expõe endpoints
para receber múltiplas imagens, aplicar um pipeline de processamento e devolver
ambas as versões (original e processada) em Base64.

Usado em:
- front-end: upload em lote e visualização do resultado (rota /process-images)
"""

import io
import base64
from typing import List, Dict, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Processing API",
    description="API para processamento de imagens em lote",
    version="1.0.0"
)

# CORS
origins = ["http://localhost:3000",]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONFIGURAÇÕES
MAX_FILE_SIZE = 10 * 1024 * 1024 # 10MB por imagem
ALLOWED_EXTENSIONS = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp'}


def validate_image(file: UploadFile) -> None:
    """
    @function validate_image
    Descrição: Valida o content-type do arquivo recebido garantindo que é um
    formato de imagem suportado pelo pipeline.

    Parâmetros:
    - file (UploadFile): Arquivo recebido via multipart/form-data com
      content_type definido pelo cliente.

    Retorna:
    - None: Somente valida.

    Exceções:
    - HTTPException: Lançada com status 400 quando o tipo não é suportado.
    """
    if file.content_type not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: {file.content_type}. Use JPEG, PNG ou WEBP."
        )

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    @function image_to_base64
    Descrição: Converte uma instância PIL.Image para string Base64 em Data URI,
    normalizando o modo de cor quando necessário para o formato alvo.

    Parâmetros:
    - image (PIL.Image.Image): Imagem em memória a ser serializada.
    - format (str): Formato de saída. Suporta "JPEG", "PNG" e "WEBP".

    Retorna:
    - str: Data URI no padrão "data:<mime>;base64,<payload>".

    Exceções:
    - ValueError/OSError: Propagadas pela PIL em falhas de conversão/salvamento.
    """
    buffered = io.BytesIO()
    
    if format == "JPEG" and image.mode in ("RGBA", "LA", "P"):
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        rgb_image.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
        image = rgb_image
    
    image.save(buffered, format=format, quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    mime_type = "image/jpeg" if format == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime_type};base64,{img_str}"


def detect_areas(image: Image.Image) -> Tuple[Image.Image, int, List[Dict]]:
    """
    @function detect_areas
    Descrição: Detecta as áreas (compartimentos) na imagem.
    
    Parâmetros:
    - image (PIL.Image.Image): Imagem original
    
    Retorna:
    - Tuple contendo:
      - Image.Image: Imagem com áreas destacadas
      - int: Número de áreas detectadas
      - List[Dict]: Lista de informações sobre cada área
    """
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Aplicar threshold adaptativo para detectar as grades
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos por área mínima
    min_area = (image.width * image.height) / 100  # pelo menos 1% da imagem
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Ordenar contornos por área e pegar os retângulos principais
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    
    # Criar imagem de saída
    output_img = img_array.copy()
    areas_info = []
    
    for idx, contour in enumerate(valid_contours[:20]):  # Limitar a 20 áreas
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Desenhar retângulo
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Adicionar número da área
        cv2.putText(output_img, f"Area {idx+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        areas_info.append({
            "area_id": idx + 1,
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area_pixels": int(area)
        })
    
    result_image = Image.fromarray(output_img)
    return result_image, len(valid_contours), areas_info


def detect_pins(image: Image.Image) -> Tuple[Image.Image, int, List[Dict]]:
    """
    @function detect_pins
    Descrição: Detecta e conta os pins (tachinhas) na imagem.
    
    Parâmetros:
    - image (PIL.Image.Image): Imagem original
    
    Retorna:
    - Tuple contendo:
      - Image.Image: Imagem com pins destacados
      - int: Número de pins detectados
      - List[Dict]: Lista de informações sobre cada pin
    """
    img_array = np.array(image.convert('RGB'))
    
    # Converter para HSV para detectar objetos amarelos
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Definir range para cor amarela
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Criar máscara para objetos amarelos
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Aplicar operações morfológicas para limpar a máscara
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos dos pins
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos por área
    min_pin_area = 50  # área mínima de um pin
    max_pin_area = 5000  # área máxima de um pin
    
    valid_pins = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_pin_area < area < max_pin_area:
            valid_pins.append(cnt)
    
    # Criar imagem de saída
    output_img = img_array.copy()
    pins_info = []
    
    for idx, contour in enumerate(valid_pins):
        # Calcular o centro do contorno
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Desenhar contorno e centro
        cv2.drawContours(output_img, [contour], -1, (255, 0, 0), 2)
        cv2.circle(output_img, (cx, cy), 5, (0, 0, 255), -1)
        
        # Adicionar número do pin
        cv2.putText(output_img, str(idx+1), (cx-10, cy-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        x, y, w, h = cv2.boundingRect(contour)
        pins_info.append({
            "pin_id": idx + 1,
            "center_x": cx,
            "center_y": cy,
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area": int(cv2.contourArea(contour))
        })
    
    result_image = Image.fromarray(output_img)
    return result_image, len(valid_pins), pins_info


def process_image(image: Image.Image) -> Dict:
    """
    @function process_image
    Descrição: Processa a imagem detectando áreas e pins.
    
    Parâmetros:
    - image (PIL.Image.Image): Imagem original
    
    Retorna:
    - Dict: Dicionário com imagens processadas e informações
    """
    # Detectar áreas
    areas_image, areas_count, areas_info = detect_areas(image.copy())
    
    # Detectar pins
    pins_image, pins_count, pins_info = detect_pins(image.copy())
    
    return {
        "areas_image": areas_image,
        "areas_count": areas_count,
        "areas_info": areas_info,
        "pins_image": pins_image,
        "pins_count": pins_count,
        "pins_info": pins_info
    }


@app.get("/")
async def root():
    """
    @function root
    Descrição: Endpoint raiz para verificação rápida do serviço e descoberta
    dos endpoints principais.

    Retorna:
    - dict: Metadados da API e caminhos relevantes.
    """
    return {
        "message": "Image Processing API está funcionando!",
        "version": "1.0.0",
        "endpoints": {
            "process_images": "/process-images/"
        }
    }


@app.post("/process-images/")
async def process_images(files: List[UploadFile] = File(...)):
    """
    @function process_images
    Descrição: Recebe uma lista de imagens, valida tipos e tamanhos, processa
    cada uma pelo pipeline e retorna originais e processadas em Base64.

    Parâmetros:
    - files (List[UploadFile]): Até 5 arquivos de imagem via multipart. Tipos
      aceitos: image/jpeg, image/jpg, image/png, image/webp. Tamanho máximo:
      10MB por arquivo.

    Retorna:
    - dict: {"processed_images": [{filename, original_image_data,
      areas_image_data, pins_image_data, areas_count, pins_count, 
      areas_info, pins_info}]}

    Exceções:
    - HTTPException 400: Nenhum arquivo, excesso de quantidade, tipo inválido
      ou tamanho acima do limite, ou erro de parsing da imagem.
    - HTTPException 500: Falha inesperada durante o processamento.
    """
    
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado")
    
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Máximo de 5 imagens por lote")
    
    processed_images_results = []
    
    logger.info(f"Processando {len(files)} imagens...")
    
    for idx, file in enumerate(files):
        try:
            validate_image(file)

            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Arquivo {file.filename} excede o tamanho máximo de 10MB"
                )

            try:
                original_image = Image.open(io.BytesIO(contents))
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Erro ao abrir a imagem {file.filename}: {str(e)}"
                )
            
            image_format = original_image.format or "JPEG"
            if image_format not in ["JPEG", "PNG", "WEBP"]:
                image_format = "JPEG"
            
            # Processar imagem
            processing_result = process_image(original_image.copy())
            
            # Converter para base64
            original_base64 = image_to_base64(original_image.copy(), format=image_format)
            areas_base64 = image_to_base64(processing_result["areas_image"], format=image_format)
            pins_base64 = image_to_base64(processing_result["pins_image"], format=image_format)

            processed_images_results.append({
                "filename": file.filename,
                "original_image_data": original_base64,
                "areas_image_data": areas_base64,
                "pins_image_data": pins_base64,
                "areas_count": processing_result["areas_count"],
                "pins_count": processing_result["pins_count"],
                "areas_info": processing_result["areas_info"],
                "pins_info": processing_result["pins_info"]
            })
            
            logger.info(f"Imagem {idx + 1}/{len(files)} processada: {file.filename} - "
                       f"Áreas: {processing_result['areas_count']}, Pins: {processing_result['pins_count']}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erro ao processar {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao processar a imagem {file.filename}: {str(e)}"
            )
    
    logger.info(f"Processamento concluído: {len(processed_images_results)} imagens")
    
    return {"processed_images": processed_images_results}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Erro não tratado: {str(exc)}")
    return {
        "detail": "Ocorreu um erro interno no servidor",
        "error": str(exc)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)