import cv2
import numpy as np
from skimage import color
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
def remove_hair(image, mask):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    backhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, hairmask = cv2.threshold(backhat, 10, 255, cv2.THRESH_BINARY)
    
    final_mask = cv2.bitwise_or(mask, hairmask)

    return final_mask

GLOBAL_MIN, GLOBAL_MAX = -90, 135
# 步长为 1°，生成从 -90.5 到 135.5 的边界
GLOBAL_BINS = np.arange(GLOBAL_MIN - 0.5, GLOBAL_MAX + 1.5, 1.0)
NUM_FEATURES = len(GLOBAL_BINS) - 1 

def cal_ita_histogram(image, mask):
    img_lab = color.rgb2lab(image)
    l_channel = img_lab[:, :, 0]
    b_channel = img_lab[:, :, 2]
    
    valid_skin = (mask == 0) & (l_channel > 10) 
    skin_pixels = img_lab[valid_skin]

    if len(skin_pixels) < (image.shape[0] * image.shape[1] * 0.01):
        return np.zeros(NUM_FEATURES), np.nan

    l, b = skin_pixels[:, 0], skin_pixels[:, 2]
    ita = np.arctan2(l - 50, b) * 180 / np.pi
    
    median_val = np.median(ita)

    # 裁减异常值，确保所有有效像素都落在 GLOBAL_BINS 范围内
    ita_clipped = np.clip(ita, GLOBAL_MIN, GLOBAL_MAX)
    
    # 生成概率密度分布 (Equation 2)
    hist, _ = np.histogram(ita_clipped, bins=GLOBAL_BINS, density=True)
    return hist, median_val

def process_image(row, img_dir, mask_dir, output_dir):
    img_id = row['image']
    
    img_path = os.path.join(img_dir, img_id + '.jpg')
    mask_path = os.path.join(mask_dir, img_id + '_segmentation.png')
    
    image_bgr = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or mask is None:
        return None
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    
    
    
    final_mask = remove_hair(image, mask)
    
    qi, median_val = cal_ita_histogram(image, final_mask)
    np.save(os.path.join(output_dir, img_id + '.npy'), qi)
    return {'image': img_id, 'median_ita': median_val}

if __name__ == '__main__':
    img_dir = '/root/skinai/data/images'
    mask_dir = '/root/skinai/data/masks'
    output_dir = '/root/skinai/data/qi'
    csv_path = '/root/skinai/data/GroundTruth.csv'
    df = pd.read_csv(csv_path)

    os.makedirs(output_dir, exist_ok=True)

   
    tasks = df.to_dict('records')

    worker_func = partial(process_image, img_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir)

    print(f"开始处理 {len(tasks)} 个任务")
    with ProcessPoolExecutor(max_workers=24) as executor:
        results = list(tqdm(executor.map(worker_func, tasks), total=len(tasks)))

    failed = [r for r in results if r is None]
    print(f"处理完成，成功 {len(results) - len(failed)} 个任务，失败 {len(failed)} 个任务")
