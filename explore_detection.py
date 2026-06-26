import json, sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

def detect_and_crop(image_path, conf=0.25, padding=10):
    model = YOLO('yolov8x.pt')
    image = Image.open(image_path).convert('RGB')
    img_w, img_h = image.size
    print(f'Image size: {img_w} x {img_h}')

    results = model(image_path, conf=conf, verbose=False)

    detections = []
    Path('data/crops').mkdir(parents=True, exist_ok=True)

    for result in results:
        for i, box in enumerate(result.boxes):
            label = result.names[int(box.cls)]
            bbox  = [round(v, 2) for v in box.xyxy[0].tolist()]
            conf_score = round(float(box.conf[0]), 4)

            x1 = max(0, int(bbox[0]) - padding)
            y1 = max(0, int(bbox[1]) - padding)
            x2 = min(img_w, int(bbox[2]) + padding)
            y2 = min(img_h, int(bbox[3]) + padding)
            crop = image.crop((x1, y1, x2, y2))
            crop_path = f'data/crops/{label}_{i}.jpg'
            crop.save(crop_path, quality=95)

            detections.append({
                'label': label,
                'bbox': bbox,
                'confidence': conf_score,
                'crop_file': crop_path
            })
            print(f'  {label:20s}  conf={conf_score:.3f}')

    with open('data/detections.json', 'w') as f:
        json.dump(detections, f, indent=2)
    print(f'Saved {len(detections)} detections → data/detections.json')
    print(f'Crops saved → data/crops/')
    return detections

if __name__ == '__main__':
    img = sys.argv[1] if len(sys.argv) > 1 else 'data/test1.jpg'
    detect_and_crop(img)