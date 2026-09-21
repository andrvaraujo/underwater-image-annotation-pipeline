import os
import cv2
import albumentations as A

INPUT_FOLDER = "images2"
OUTPUT_FOLDER = "augmented_images"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

NUM_AUG_PER_IMAGE = 2  #versoespor imagem

transform = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(-0.02, 0.04),
        contrast_limit=(0.08,0.15),
        p = 0.6
    ),


    A.Sharpen(alpha=(0.1, 0.25), lightness=(0.9, 1.1), p=0.7),

    A.GaussianBlur(blur_limit=(2, 4), p=0.0),

    A.GaussNoise(std_range=(0.01, 0.3), p=0.0),
 
     A.RGBShift(
        r_shift_limit=(-0.08, -0.05),
        g_shift_limit=(0.0, 0.0),
        b_shift_limit=(0.0, 0.0),
        p=0.4
    ),

    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.0
    ),

    # contraste local 
    A.CLAHE(clip_limit=(1.5, 2.5), tile_grid_size=(8,8), p=0.6),

    #preto e branco
    A.ToGray(p=0.0),

])



print("a começar")

for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(path)

    if image is None:
        continue

    name, ext = os.path.splitext(filename)

    # guardar original 
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{name}_orig{ext}"), image)

    # gerar versões augmentadas
    for i in range(NUM_AUG_PER_IMAGE):
        augmented = transform(image=image)
        aug_img = augmented["image"]

        save_path = os.path.join(OUTPUT_FOLDER, f"{name}_aug_{i}{ext}")
        cv2.imwrite(save_path, aug_img)

print("terminado.")