import os
import cv2
import numpy as np
from pathlib import Path

def process_folder(
        input_dir='C:/Users/evand/OneDrive/Documentos/Meus Programas/Analisador/HawkEye-BackEnd/test/imagens_pins', output_dir='C:/Users/evand/OneDrive/Documentos/Meus Programas/Analisador/HawkEye-BackEnd/test/identificar-pins/processados/',
        lower_yellow=(10, 155, 100), 
        upper_yellow=(30, 255, 255),
        min_area=2000):
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.png', '.jpg', '.jpeg',}
    lower = np.array(lower_yellow, dtype=np.uint8)
    upper = np.array(upper_yellow, dtype=np.uint8)

    processed = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                print(f"Falha ao ler: {p.name} — ignorando")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = [c for c in contours if cv2.contourArea(c) > min_area]

            annotated = img_rgb.copy()
            cv2.drawContours(annotated, filtered, -1, (255, 0, 255), 3)

            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

            out_ann = output_dir / f"{p.stem}_resultado{p.suffix}"
            cv2.imwrite(str(out_ann), annotated_bgr)

            processed.append((p.name, out_ann.name, len(filtered)))

    return processed

if __name__ == '__main__':
    summary = process_folder()
    if summary:
        print("Processamento concluído. Arquivos salvos em: 'processados")
        print("original -> nº contornos")
        for orig, ann, n in summary:
            print(f"{orig} -> {n} contornos")
    else:
        print("Nenhuma imagem processada.")