import os
import cv2
import torch
from PIL import Image
from rembg import remove
import numpy as np

# Make sure you have downloaded GFPGANv1.3.pth from: https://github.com/TencentARC/GFPGAN
from gfpgan import GFPGANer

# Initialize GFPGAN
gfpganer = GFPGANer(
    model_path='GFPGANv1.3.pth', upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None
)

# Load your input image
input_path = 'your_photo.jpg'  # 🔁 Replace this with your photo filename
input_image = cv2.imread(input_path)

# Step 1: Enhance face
cropped_face, _, restored_img = gfpganer.enhance(input_image, has_aligned=False, only_center_face=False)
cv2.imwrite("enhanced_face.jpg", restored_img)

# Step 2: Remove background
input_pil = Image.open("enhanced_face.jpg").convert("RGBA")
output_pil = remove(input_pil)
output_pil.save("face_nobg.png")

# Step 3: Add background (replace with your own sky/beach image)
bg = Image.open("artistic_background.jpg").resize(output_pil.size).convert("RGBA")
final = Image.alpha_composite(bg, output_pil)

# Step 4: Save final
final.save("final_effect_result.png")
print("✅ Stylized photo saved as final_effect_result.png")
