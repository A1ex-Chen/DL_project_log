import os
import urllib.request

import gdown

if __name__ == "__main__":
    datasets_dir: str = "datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    
    urllib.request.urlretrieve("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", os.path.join(f"imagenette2.tgz"))
    urllib.request.urlretrieve("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", os.path.join(f"imagenette2-320.tgz"))
    
    imagenette_dir: str = "real_images/imagenette_256"
    os.makedirs(imagenette_dir, exist_ok=True)
    
    os.system("python ffhq-dataset/download_ffhq.py -i --source-dir datasets/ffhq")