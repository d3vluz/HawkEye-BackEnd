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
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import logging

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

def process_image(image: Image.Image) -> Image.Image:
    """
    Em Desenvolvimento: Função de processamento de imagem.
    
    Aqui será aplicado todas as transformações necessárias.
    """
    # Teste de Validação: Converter para escala de cinza
    processed = image.convert("L")
    
    return processed

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
      processed_image_data}]}

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
            
            original_base64 = image_to_base64(original_image.copy(), format=image_format)
            processed_image = process_image(original_image.copy())
            processed_base64 = image_to_base64(processed_image, format=image_format)

            processed_images_results.append({
                "filename": file.filename,
                "original_image_data": original_base64,
                "processed_image_data": processed_base64
            })
            
            logger.info(f"Imagem {idx + 1}/{len(files)} processada com sucesso: {file.filename}")
            
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