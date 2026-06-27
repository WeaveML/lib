#!/usr/bin/env python3
import sys
from PIL import Image

def convert_to_raw(image_path, output_path):
    img = Image.open(image_path).convert('L')
    
    if img.size != (100, 100):
        img = img.resize((100, 100))
        
    with open(output_path, 'wb') as f:
        f.write(img.tobytes())

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("use: python digit_gen.py input.png output.raw")
    else:
        convert_to_raw(sys.argv[1], sys.argv[2])
