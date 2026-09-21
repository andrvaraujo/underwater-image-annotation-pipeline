import os
import cv2
import json
import numpy as np
import requests
from tqdm import tqdm
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
import torch
import clip
from PIL import Image
import gc

#configuracao inicial

IMAGE_FOLDER = "images"
CHECKPOINT = "mobile_sam.pt"

CVAT_URL = os.getenv("CVAT_URL", "http://localhost:8080")
USERNAME = os.getenv("CVAT_USERNAME")
PASSWORD = os.getenv("CVAT_PASSWORD")

if not USERNAME or not PASSWORD:
    raise RuntimeError(
        "Set CVAT_USERNAME and CVAT_PASSWORD environment variables."
    )

MIN_AREA = 1000

#carregar modelo SAM

print("Loading MobileSAM")
sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=12,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.75,
    min_mask_region_area=600
)

##CLIP

print("Loading CLIP")

device = "cpu"

clip_model, preprocess = clip.load(
    "ViT-B/32",
    device=device
)

LABELS = [
    "fish",
    "robot",
    "plastic cup",
    "torpedo AUV",
    "scissors",
    "fork",
    "car",
    "goggles",
    "screwdriver",
    "tin can",
    "rock",
    "plastic bottle",
    "rope",
    "diver equipment",
    "toy fish",
    "cone",
]
ALL_LABELS = LABELS + ["unknown object"]

text_inputs = torch.cat([
    clip.tokenize(f"an underwater photo of a {label}")
    for label in LABELS
]).to(device)

LABEL_TO_ID = {
    label: i + 1
    for i, label in enumerate(ALL_LABELS)
}


#mascaras

def mask_to_polygons(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        contour = contour.squeeze()

        if len(contour.shape) != 2:
            continue

        if len(contour) >= 3:
            approx = cv2.approxPolyDP(contour, 1.0, True)
            polygons.append(approx.squeeze().tolist())

    return polygons


def polygon_to_bbox(poly):
    poly = np.array(poly)
    x_min = np.min(poly[:, 0])
    y_min = np.min(poly[:, 1])
    x_max = np.max(poly[:, 0])
    y_max = np.max(poly[:, 1])

    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def classify_crop(crop_bgr):

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(crop_rgb)

    image_input = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():

        logits_per_image, logits_per_text = clip_model(
            image_input,
            text_inputs
        )

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    best_idx = np.argmax(probs)

    label = LABELS[best_idx]
    confidence = probs[best_idx]

    return label, confidence


#criar task no cvat

print("Creating CVAT task")

task_data = {
    "name": "SAM Auto Task",
    "labels": [
        {"name": label}
        for label in ALL_LABELS
    ]
}

r = requests.post(f"{CVAT_URL}/api/tasks", json=task_data, auth=(USERNAME, PASSWORD))

if r.status_code != 201:
    raise Exception(r.text)

task_id = r.json()["id"]
print(f"Task ID: {task_id}")


#upload das imagens
 
print("Uploading images")

files = {}
image_list = []

for i, filename in enumerate(os.listdir(IMAGE_FOLDER)):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(IMAGE_FOLDER, filename)
        files[f'client_files[{i}]'] = (filename, open(path, 'rb'))
        image_list.append(filename)

r = requests.post(
    f"{CVAT_URL}/api/tasks/{task_id}/data",
    files=files,
    data={"image_quality": 70},
    auth=(USERNAME, PASSWORD)
)

if r.status_code not in [200, 202]:
    raise Exception(r.text)

print("Images uploaded.")
for f in files.values():
    f[1].close()


#gerar anotacoes

coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": idx,
            "name": label
        }
        for label, idx in LABEL_TO_ID.items()
    ]
}

annotation_id = 1
image_id = 1

print("Running SAM...")

for filename in tqdm(image_list):

    path = os.path.join(IMAGE_FOLDER, filename)
    image = cv2.imread(path)

    if image is None:
        continue

    h_orig, w_orig = image.shape[:2]

    image = cv2.resize(image, (512, 512))
    scale_x = w_orig / 512
    scale_y = h_orig / 512
    

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("mascaras a serem geradas")
    masks = mask_generator.generate(image_rgb)
    print(f"geradas {len(masks)} mascaras")

    MAX_AREA_RATIO = 0.35

    filtered_masks = []
    for m in masks:
        area_ratio = m["area"] / (512 * 512)
        
        if area_ratio > MAX_AREA_RATIO:
            continue
        
        filtered_masks.append(m)

    masks = filtered_masks

    
    masks = sorted(masks, key=lambda x: x["predicted_iou"], reverse=True)[:5]
    

    coco["images"].append({
        "id": image_id,
        "file_name": filename,
        "width": w_orig,
        "height": h_orig
    })

    for m in masks:
        
        seg = m["segmentation"]
        ys, xs = np.where(seg)

        if len(xs) == 0 or len(ys) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        w = x2 - x1
        h = y2 - y1

        
        if h == 0 or w == 0:
            continue

        
        aspect_ratio = max(w, h) / min(w, h)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        touches_border = (
            x1 <= 10 or
            y1 <= 10 or
            x2 >= 512 - 10 or
            y2 >= 512 - 10
        )

        if touches_border:
            continue

        if aspect_ratio > 8:
            continue

        if center_y > 512 * 0.85:
            continue

        masked_image = image.copy()

        masked_image[seg == 0] = 0

        crop = masked_image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        label, confidence = classify_crop(crop)
        del crop
        gc.collect()

        print(f"{filename} --> {label} ({confidence:.2f})")

        if confidence < 0.40:
            label = "unknown object"
        
        if m["area"] < MIN_AREA:
            continue

        polygons = mask_to_polygons(m["segmentation"])

        for poly in polygons:
            if len(poly) < 3:
                continue

            poly = np.array(poly).astype(float)

            poly[:, 0] = poly[:, 0] * scale_x
            poly[:, 1] = poly[:, 1] * scale_y

            segmentation = [poly.flatten().tolist()]
            bbox = polygon_to_bbox(poly)

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": LABEL_TO_ID[label],
                "segmentation": segmentation,
                "bbox": bbox,
                "area": float(m["area"] * scale_x * scale_y),
                "iscrowd": 0
            })

            annotation_id += 1

    image_id += 1


#guardar json 

with open("annotations.json", "w") as f:
    json.dump(coco, f)

print("Annotations saved.")


#enviar para o cvat 

print("Uploading annotations")

with open("annotations.json", "rb") as f:
    r = requests.post(
        f"{CVAT_URL}/api/tasks/{task_id}/annotations?format=COCO%201.0",
        files={"annotation_file": f},
        auth=(USERNAME, PASSWORD)
    )

if r.status_code not in [200, 202]:
    raise Exception(r.text)

print("FEITO. Depois desta a NASA pode me contratar.")
