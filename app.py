from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
from ultralytics import YOLO
from typing import List

app = FastAPI()

# Создаем папку для сохранения изображений
os.makedirs("output_images", exist_ok=True)

# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8n-seg.pt")

@app.post("/detect_vehicles/")
async def detect_vehicles(file: UploadFile = File(...)) -> JSONResponse:
    # Чтение входного изображения
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # Выполнение инференса модели
    results = model(image)

    output_images = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Координаты bounding box
        masks = result.masks.data.cpu().numpy()  # Маски объектов
        class_ids = result.boxes.cls.cpu().numpy()  # ID классов
        names = result.names  # Словарь с именами классов

        for i, (box, mask, cls_id) in enumerate(zip(boxes, masks, class_ids)):
            # Проверяем, является ли объект средством передвижения
            if names[int(cls_id)] not in ["car", "bicycle", "motorcycle", "bus", "truck"]:
                continue

            # Изменяем размер маски до размера исходного изображения
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

            # Преобразуем маску в формат uint8
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            # Применяем маску к изображению
            masked_image = cv2.bitwise_and(image, image, mask=mask_uint8)

            # Добавляем подпись
            pil_image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.load_default()
            draw.text((10, 10), names[int(cls_id)], fill="red", font=font)

            # Сохраняем изображение в папку
            output_filename = f"output_images/{names[int(cls_id)]}_{i}.png"
            pil_image.save(output_filename)

            # Добавляем путь к сохраненному изображению в результат
            output_images.append({
                "class": names[int(cls_id)],
                "filename": output_filename
            })

    return JSONResponse(content=output_images)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)