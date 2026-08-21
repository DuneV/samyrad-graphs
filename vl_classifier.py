import json
from pathlib import Path
import cv2
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

def extract_frames(video_path: str, num_frames: int = 4) -> list[Image.Image]:
    """Extrae N frames distribuidos equitativamente a lo largo de un archivo de video MP4."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el archivo de video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"El video {video_path} no tiene frames válidos.")

    # Calcular los índices de los frames distribuidos uniformemente
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    images = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV carga en BGR; convertimos a RGB para Pillow/PyTorch
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame_rgb))

    cap.release()

    if len(images) == 0:
        raise RuntimeError(f"No se pudieron extraer frames de {video_path}")

    return images


def main():
    # 1. Configurar ruta del video de EPIC-KITCHENS
    # Modifica esta ruta según el video específico que desees analizar
    video_path = Path("kitchen/EPIC-KITCHENS/P01/videos/P01_01.MP4")

    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el video en la ruta: {video_path.resolve()}")

    print(f"[+] Extrayendo 4 frames de: {video_path}")
    images = extract_frames(str(video_path), num_frames=4)

    # 2. Cargar modelo Qwen2-VL-7B en 4-bit (ocupa ~6.5 - 7.5 GB VRAM)
    # 2. Cargar modelo Qwen2-VL-7B en 4-bit mediante BitsAndBytesConfig
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    print(f"[+] Cargando modelo {model_id} en 4-bit...")

    # Configuración explícita de cuantización a 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # 3. Diseñar prompt estricto para salida en tripletas JSON
    prompt = """Analiza la secuencia temporal de estas 4 imágenes del video de cocina.
Identifica la acción principal realizada y extrae las tripletas semánticas (Sujeto, Acción, Objeto Objetivo).

Responde ÚNICAMENTE con una lista JSON válida y sin texto adicional, siguiendo este formato exacto:
[
  {
    "frame_sequence": "1-4",
    "subject": "mano derecha",
    "action": "tomar",
    "target_object": "vaso rojo"
  }
]
"""

    # 4. Construir la estructura del mensaje para el procesador
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # 5. Preprocesar las entradas visuales y de texto
    text_prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text_prompt,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # 6. Generar respuesta
    print("[+] Ejecutando inferencia...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1  # Temperatura baja para maximizar determinismo e integridad del JSON
        )

    # Recortar los tokens del prompt original de la salida
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print("\n=== Resultado Salida JSON ===")
    print(output_text)


if __name__ == "__main__":
    main()