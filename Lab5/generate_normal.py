import os
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import filters

def generate_normal_patches():
    image_path = "C:/Users/masad/Desktop/R.jfif"
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return
        
    img = Image.open(image_path).convert("L")
    gray = np.asarray(img)
    
    # Calculate edges to find plain areas
    blurred = filters.gaussian(gray, sigma=2.0)
    edges = filters.sobel(blurred)
    
    # Find a threshold for edges
    # We want patches where the maximum edge response is very low
    
    out_dir = Path("c:/Users/masad/Desktop/computer_vi/Lab5/normal_images")
    out_dir.mkdir(exist_ok=True)
    
    patch_size = 80
    h, w = gray.shape
    stride = 80
    
    count = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            edge_patch = edges[y:y+patch_size, x:x+patch_size]
            if np.max(edge_patch) < 0.05: # very low edge energy -> smooth steel
                patch = gray[y:y+patch_size, x:x+patch_size]
                patch_img = Image.fromarray(patch)
                patch_img.save(out_dir / f"normal_{count}.jpg")
                count += 1
                if count >= 300: # Need enough normal images to balance the dataset
                    break
        if count >= 300:
            break
            
    print(f"Saved {count} normal patches.")

if __name__ == '__main__':
    generate_normal_patches()
