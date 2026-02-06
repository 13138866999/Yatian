import cv2
import numpy as np
from skimage import color
import os
import pandas as pd
import logging
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
def _get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _normalize_path(path, base_dir):
    if path is None:
        return None
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))

def _resolve_paths(csv_path=None, output_dir=None, img_dir=None, mask_dir=None, meta_path=None, base_dir=None):
    base_dir = os.path.abspath(base_dir or os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    if csv_path is None:
        csv_path = os.path.join(data_dir, 'GroundTruth.csv')
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'qi')
    if img_dir is None:
        img_dir = os.path.join(data_dir, 'images')
    if mask_dir is None:
        mask_dir = os.path.join(data_dir, 'masks')
    if meta_path is None:
        meta_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    return {
        'base_dir': base_dir,
        'data_dir': data_dir,
        'csv_path': _normalize_path(csv_path, base_dir),
        'output_dir': _normalize_path(output_dir, base_dir),
        'img_dir': _normalize_path(img_dir, base_dir),
        'mask_dir': _normalize_path(mask_dir, base_dir),
        'meta_path': _normalize_path(meta_path, base_dir)
    }

def _get_logger(logger=None):
    if logger is not None:
        return logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("preprocessing")

def _merge_config(config):
    default_config = {
        'steps': ['load_raw', 'diagnosis', 'merge_ita', 'merge_meta', 'clean', 'validate'],
        'drop_duplicates': True,
        'drop_missing': True,
        'require_meta': True,
        'merge_how': 'inner'
    }
    if config is None:
        return default_config
    merged = default_config.copy()
    merged.update(config)
    return merged

def _validate_columns(df, required_cols, logger):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error("missing_required_columns=%s", missing)
        raise ValueError(f"Missing required columns: {missing}")

def _normalize_diagnosis(df, logger):
    disease_columns = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    existing_disease_cols = [col for col in disease_columns if col in df.columns]
    if len(existing_disease_cols) > 0:
        def get_diagnosis(row):
            for col in existing_disease_cols:
                if row[col] == 1.0:
                    return col
            return 'Unknown'
        df = df.copy()
        df['diagnosis'] = df.apply(get_diagnosis, axis=1)
        df = df.drop(columns=existing_disease_cols)
        logger.info("diagnosis_columns_removed=%s", existing_disease_cols)
    else:
        if 'diagnosis' in df.columns:
            df = df.copy()
            df['diagnosis'] = df['diagnosis'].astype(str)
        elif 'dx' in df.columns:
            df = df.copy()
            df['diagnosis'] = df['dx'].astype(str)
        else:
            df = df.copy()
            df['diagnosis'] = 'Unknown'
        logger.info("diagnosis_columns_missing=True")
    return df

def _merge_ita(df, output_dir, logger, merge_how):
    ita_csv_path = os.path.join(output_dir, 'ita_medians.csv')
    if not os.path.exists(ita_csv_path):
        logger.warning("ita_medians_missing=%s", ita_csv_path)
        return df
    ita_df = pd.read_csv(ita_csv_path)
    ita_df = ita_df.dropna(subset=['median_ita'])
    merged = pd.merge(df, ita_df, on='image', how=merge_how)
    logger.info("merge_ita_shape=%s", merged.shape)
    return merged

def _merge_metadata(df, meta_path, require_meta, logger):
    if not os.path.exists(meta_path):
        if require_meta:
            raise FileNotFoundError(f"CRITICAL: {meta_path} missing! Cannot perform lesion-level split.")
        logger.warning("metadata_missing=%s", meta_path)
        return df
    meta = pd.read_csv(meta_path, usecols=['lesion_id', 'image_id'])
    merged = pd.merge(df, meta, left_on='image', right_on='image_id', how='left')
    merged = merged.drop(columns=['image_id'])
    logger.info("merge_meta_shape=%s", merged.shape)
    return merged

def _clean_data(df, drop_duplicates, drop_missing, logger):
    cleaned = df
    if drop_duplicates and 'image' in cleaned.columns:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=['image'])
        logger.info("drop_duplicates=%s", before - len(cleaned))
    if drop_missing:
        before = len(cleaned)
        cleaned = cleaned.dropna()
        logger.info("drop_missing=%s", before - len(cleaned))
    return cleaned

def prepare_data(df=None, csv_path=None, output_dir=None, meta_path=None, config=None, config_path=None, base_dir=None, logger=None):
    paths = _resolve_paths(csv_path=csv_path, output_dir=output_dir, meta_path=meta_path, base_dir=base_dir)
    csv_path = paths['csv_path']
    output_dir = paths['output_dir']
    meta_path = paths['meta_path']
    logger = _get_logger(logger)
    if config_path is not None:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    cfg = _merge_config(config)
    steps = cfg.get('steps', [])
    if df is None and 'load_raw' in steps:
        df = pd.read_csv(csv_path)
        logger.info("load_raw_shape=%s", df.shape)
    if df is None:
        raise ValueError("No input dataframe or csv_path provided")
    if 'diagnosis' in steps:
        df = _normalize_diagnosis(df, logger)
    if 'merge_ita' in steps:
        df = _merge_ita(df, output_dir, logger, cfg.get('merge_how', 'inner'))
    if 'merge_meta' in steps:
        df = _merge_metadata(df, meta_path, cfg.get('require_meta', True), logger)
    if 'clean' in steps:
        df = _clean_data(df, cfg.get('drop_duplicates', True), cfg.get('drop_missing', True), logger)
    if 'validate' in steps:
        _validate_columns(df, ['image', 'diagnosis'], logger)
    return df
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
    paths = _resolve_paths()
    img_dir = paths['img_dir']
    mask_dir = paths['mask_dir']
    output_dir = paths['output_dir']
    csv_path = paths['csv_path']
    df = pd.read_csv(csv_path)

    os.makedirs(output_dir, exist_ok=True)

   
    tasks = df.to_dict('records')

    worker_func = partial(process_image, img_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir)

    print(f"开始处理 {len(tasks)} 个任务")
    with ProcessPoolExecutor(max_workers=24) as executor:
        results = list(tqdm(executor.map(worker_func, tasks), total=len(tasks)))

    failed = [r for r in results if r is None]
    print(f"处理完成，成功 {len(results) - len(failed)} 个任务，失败 {len(failed)} 个任务")
