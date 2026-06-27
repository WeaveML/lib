#!/usr/bin/env python3
import os
import random
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
import cv2

FONTS_DIR = "fonts"
OUTPUT_DIR = "generated_dataset"
IMAGE_SIZE = (100, 100)

os.makedirs(OUTPUT_DIR, exist_ok=True)

fonts_file = open(FONTS_DIR+"/fonts")
font_paths = fonts_file.read()

if not font_paths:
    raise FileNotFoundError(f"No fonts found in '{FONTS_DIR}'.")

transform_pipeline = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.12,
        scale_limit=0.1,
        rotate_limit=20,
        border_mode=cv2.BORDER_CONSTANT,
        value=255,
        p=0.9
    ),
    A.ElasticTransform(
        alpha=1,
        sigma=50,
        alpha_affine=50,
        border_mode=cv2.BORDER_CONSTANT,
        value=255,
        p=0.4
    ),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.Blur(blur_limit=3),
    ], p=0.4),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 60.0)),
        A.PixelDropout(dropout_prob=0.05, per_channel=False),
    ], p=0.5)
])

def generate_base_digit(digit, font_path, size):
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    
    font_size = random.randint(int(size[0] * 0.6), int(size[0] * 0.8))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        return None

    text = str(digit)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (size[0] - text_width) // 2 - text_bbox[0] + random.randint(-2, 2)
    y = (size[1] - text_height) // 2 - text_bbox[1] + random.randint(-2, 2)
    
    draw.text((x, y), text, fill=0, font=font)
    return img

def create_dataset(count_per_digit=100):
    print(f"Starting dataset generation. Fonts discovered: {len(font_paths)}")
    
    generated_count = 0
    for digit in range(10):
        digit_dir = os.path.join(OUTPUT_DIR, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
        
        for i in range(count_per_digit):
            font_path = random.choice(font_paths)
            
            pil_img = generate_base_digit(digit, font_path, IMAGE_SIZE)
            if pil_img is None:
                continue
                
            img_array = np.array(pil_img)
            
            augmented = transform_pipeline(image=img_array)
            corrupted_img_array = augmented['image']
            
            filename = f"digit_{digit}_{i}_{random.randint(100,999)}.png"
            filepath = os.path.join(digit_dir, filename)
            
            cv2.imwrite(filepath, corrupted_img_array)
            generated_count += 1
            
    print(f"Successfully generated {generated_count} images in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    create_dataset(count_per_digit=50)
