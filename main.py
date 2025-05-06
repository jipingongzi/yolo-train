from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11l-seg.pt")
model.train(data="carparts-seg.yaml", epochs=1, imgsz=640)
results = model.predict(source="car.png")
print(model.names)
for r in results:
    img = np.copy(r.orig_img)
    img_name = Path(r.path).stem
    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]
        b_mask = np.zeros(img.shape[:2], np.uint8)
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        isolated = np.dstack([img, b_mask])
        x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        iso_crop = isolated[y1:y2, x1:x2]
        cv2.imwrite(f"{img_name}_{label}-{ci}.png", iso_crop)
