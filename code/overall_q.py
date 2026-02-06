from typing import Any
import argparse
import time
from preprocessing import process_image, prepare_data
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from scipy.stats import gaussian_kde
from skin import SkinDs
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score
import random

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

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def get_skin_tone_label(ita):
    """
    Type 1 (Very Light): ITA > 55
    Type 2 (Light): 41 < ITA <= 55
    Type 3 (Intermediate): 28 < ITA <= 41
    Type 4 (Tan): 10 < ITA <= 28
    Type 5 (Brown): -30 < ITA <= 10
    Type 6 (Dark): ITA <= -30
    """
    if ita > 55: return 'Type 1'
    elif ita > 41: return 'Type 2'
    elif ita > 28: return 'Type 3'
    elif ita > 10: return 'Type 4'
    elif ita > -30: return 'Type 5'
    else: return 'Type 6'

def preprocessing(img_dir=None, mask_dir=None, output_dir=None, dataf=None, csv_path=None, base_dir=None):
    paths = _resolve_paths(csv_path=csv_path, output_dir=output_dir, img_dir=img_dir, mask_dir=mask_dir, base_dir=base_dir)
    img_dir = paths['img_dir']
    mask_dir = paths['mask_dir']
    output_dir = paths['output_dir']
    csv_path = paths['csv_path']
    if dataf is None:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"CSV file not found at {csv_path} and no DataFrame provided.")
    else:
        df = dataf
        
    os.makedirs(output_dir, exist_ok=True)

   
    tasks = df.to_dict('records')

    worker_func = partial(process_image, img_dir=img_dir, mask_dir=mask_dir, output_dir=output_dir)

    print(f"开始处理 {len(tasks)} 个任务")
    with ProcessPoolExecutor(max_workers=24) as executor:
        results = list(tqdm(executor.map(worker_func, tasks), total=len(tasks)))

    valid_results = [r for r in results if r is not None]
    failed = [r for r in results if r is None]
    
    if valid_results:
        medians_df = pd.DataFrame(valid_results)
        medians_csv_path = os.path.join(output_dir, 'ita_medians.csv')
        medians_df.to_csv(medians_csv_path, index=False)
        print(f"Saved median ITA values to {medians_csv_path}")

    print(f"处理完成，成功 {len(valid_results)} 个任务，失败 {len(failed)} 个任务")
    
def calOverallQ(df, mode='bs', qi_dir=None, output_dir=None, base_dir=None):
    methods = ['fs', 'wd', 'pf', 'bs']
    if mode not in methods:
        raise ValueError(f"Mode {mode} not in {methods}")
    base_dir = _get_base_dir() if base_dir is None else base_dir
    if qi_dir is None:
        qi_dir = output_dir or os.path.join(base_dir, 'data', 'qi')
    qi_dir = _normalize_path(qi_dir, base_dir)
    
    train_histograms = []
    missing_files = []
    
    for img_id in df['image']:
        file_path = os.path.join(qi_dir, f'{img_id}.npy')
        try:
            if os.path.exists(file_path):
                qi = np.load(file_path)
                train_histograms.append(qi)
            else:
                missing_files.append(file_path)
        except Exception as e:
            missing_files.append(file_path)
    
    if not train_histograms:
        raise ValueError(f"No valid histogram files found. Missing files: {missing_files[:5]}...")
    
    # Calculate reference as median of all valid histograms
    Q_ref = np.median(np.array(train_histograms), axis=0)
    q_ref_norm = Q_ref / (np.sum(Q_ref) + 1e-12)

    distances = []
    for q_i in train_histograms:
        q_i_norm = q_i / (np.sum(q_i) + 1e-12)
        if mode == 'fs':
            fs_similarity = np.sum(np.sqrt(q_i_norm * q_ref_norm))
            distances.append(1.0 - fs_similarity)
        elif mode == 'wd':
            cdf_i = np.cumsum(q_i_norm)
            cdf_ref = np.cumsum(q_ref_norm)
            wd_distance = np.sum(np.abs(cdf_i - cdf_ref))
            distances.append(wd_distance)
        elif mode == 'pf':
            pf_similarity = np.sqrt(np.sum((q_i_norm - q_ref_norm) * (q_i_norm - q_ref_norm)))
            distances.append(pf_similarity)
        elif mode == 'bs':
            bs_similarity = np.sum(np.abs(q_i_norm - q_ref_norm))
            distances.append(bs_similarity)
    return q_ref_norm, distances


def calculate_drw_weights(distances):
    kde = gaussian_kde(distances)
    density = kde(distances)
    
    d_min = np.min(density)
    d_max = np.max(density)
    
    if d_max - d_min < 1e-8:
        return np.ones_like(distances)
    
    weights = 1.0 - (density - d_min) / (d_max - d_min)
    return weights

def evaluate(model, data_loader, criterion=None):
    model.eval()
    all_predictions = []
    all_labels = []
    all_weights = []
    all_skin_tones = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.inference_mode():
        for images, labels, weights, skin_tones in data_loader:
            images = images.to('cuda', non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to('cuda', non_blocking=True)
            
            outputs = model(images)
            
            if criterion is not None:
                raw_loss_per_sample = criterion(outputs, labels)
                total_loss += raw_loss_per_sample.sum().item()
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_weights.extend(weights.cpu().numpy())
            all_skin_tones.extend(list(skin_tones))
            total_samples += labels.size(0)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    overall_f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    overall_f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    per_group_rows = []
    unique_tones = sorted(set(all_skin_tones))
    for tone in unique_tones:
        indices = [i for i, t in enumerate(all_skin_tones) if t == tone]
        if not indices:
            continue
        labels_t = [all_labels[i] for i in indices]
        preds_t = [all_predictions[i] for i in indices]
        support = len(labels_t)
        acc = accuracy_score(labels_t, preds_t)
        f1_weighted = f1_score(labels_t, preds_t, average='weighted', zero_division=0)
        f1_macro = f1_score(labels_t, preds_t, average='macro', zero_division=0)
        per_group_rows.append({
            'skin_tone': tone,
            'support': support,
            'accuracy': acc,
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro)
        })
    per_group_df = pd.DataFrame(per_group_rows)
    avg_loss = total_loss / total_samples if criterion is not None else 0.0
    
    res_df = pd.DataFrame({
        'label': all_labels,
        'prediction': all_predictions,
        'dis_weight': all_weights,
        'skin_tone': all_skin_tones,
        'correct': [l == p for l, p in zip(all_labels, all_predictions)]
    })
    
    return avg_loss, accuracy, overall_f1_macro, overall_f1_weighted, per_group_df, res_df

def _get_results_root(output_dir, mode):
    return os.path.join(os.path.dirname(output_dir), 'results', mode)

def save_evaluation(output_dir, mode, fold_index, epoch, train_loss, val_loss, val_accuracy, val_f1_macro, val_f1_weighted, per_group_df):
    results_root = _get_results_root(output_dir, mode)
    results_dir = os.path.join(results_root, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'fold_{fold_index}_{mode}_metrics.csv')
    row = pd.DataFrame([{
        'mode': mode,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted
    }])
    row.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)
    
    per_group_file = os.path.join(results_dir, f'fold_{fold_index}_{mode}_per_skin_tone.csv')
    per_group_rows = per_group_df.copy()
    per_group_rows.insert(0, 'mode', mode)
    per_group_rows.insert(0, 'epoch', epoch)
    per_group_rows.to_csv(per_group_file, mode='a', header=not os.path.exists(per_group_file), index=False)

def evaluate_test_models(df_test, label_map, output_dir, mode, batch_size, num_workers, pin_memory, persistent_workers, num_folds, image_root=None):
    results_root = _get_results_root(output_dir, mode)
    results_dir = os.path.join(results_root, 'testing')
    os.makedirs(results_dir, exist_ok=True)
    skin_ds_test = SkinDs(df_test, label_map, transform=val_transform, image_root=image_root)
    test_loader = DataLoader(
        skin_ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    fold_rows = []
    per_tone_rows = []
    for fold_index in range(num_folds):
        checkpoint_path = os.path.join(results_root, 'checkpoints', f'fold_{fold_index}_{mode}_best.pth')
        if not os.path.exists(checkpoint_path):
            continue
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, len(label_map))
        model = model.to('cuda', memory_format=torch.channels_last)
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        test_loss, test_accuracy, test_f1_macro, test_f1_weighted, per_group_df, _ = evaluate(model, test_loader, criterion=None)
        fold_rows.append({
            'mode': mode,
            'fold': fold_index,
            'test_accuracy': test_accuracy,
            'test_f1_macro': test_f1_macro,
            'test_f1_weighted': test_f1_weighted
        })
        per_group_df = per_group_df.copy()
        per_group_df.insert(0, 'mode', mode)
        per_group_df.insert(0, 'fold', fold_index)
        per_tone_rows.append(per_group_df)
    fold_df = pd.DataFrame(fold_rows)
    per_tone_df = pd.concat(per_tone_rows, ignore_index=True) if per_tone_rows else pd.DataFrame()
    fold_df.to_csv(os.path.join(results_dir, f'testing_per_fold_{mode}.csv'), index=False)
    per_tone_df.to_csv(os.path.join(results_dir, f'testing_per_skin_tone_{mode}.csv'), index=False)
    avg_overall_df = pd.DataFrame([{
        'mode': mode,
        'test_accuracy_avg': fold_df['test_accuracy'].mean() if not fold_df.empty else 0.0,
        'test_f1_macro_avg': fold_df['test_f1_macro'].mean() if not fold_df.empty else 0.0,
        'test_f1_weighted_avg': fold_df['test_f1_weighted'].mean() if not fold_df.empty else 0.0
    }])
    avg_overall_df.to_csv(os.path.join(results_dir, f'testing_avg_overall_{mode}.csv'), index=False)
    if not per_tone_df.empty:
        tone_counts = df_test['skin_tone'].value_counts().to_dict() if 'skin_tone' in df_test.columns else {}
        def _weighted_avg(group, col):
            w = group['support']
            v = group[col]
            return float((v * w).sum() / w.sum()) if w.sum() > 0 else 0.0
        rows = []
        for tone, g in per_tone_df.groupby('skin_tone'):
            rows.append({
                'mode': mode,
                'skin_tone': tone,
                'support_total': int(tone_counts.get(tone, g['support'].iloc[0] if len(g) > 0 else 0)),
                'accuracy_avg': _weighted_avg(g, 'accuracy'),
                'f1_weighted_avg': _weighted_avg(g, 'f1_weighted'),
                'f1_macro_avg': _weighted_avg(g, 'f1_macro')
            })
        avg_per_tone_df = pd.DataFrame(rows)
        avg_per_tone_path = os.path.join(results_dir, f'testing_avg_per_skin_tone_{mode}.csv')
        avg_per_tone_df.to_csv(avg_per_tone_path, index=False)
        print("Testing avg per skin tone (weighted by support):")
        print(avg_per_tone_df)
    return fold_df, per_tone_df

def run_testing_only(csv_path=None, output_dir=None, img_dir=None, mode='fs', batch_size=256, num_workers=None, pin_memory=True, base_dir=None, preprocess_config=None, preprocess_config_path=None, seed=42):
    paths = _resolve_paths(csv_path=csv_path, output_dir=output_dir, img_dir=img_dir, base_dir=base_dir)
    csv_path = paths['csv_path']
    output_dir = paths['output_dir']
    image_root = paths['img_dir']
    df = prepare_data(None, csv_path, output_dir, base_dir=base_dir, config=preprocess_config, config_path=preprocess_config_path)
    label_map = {diag: i for i, diag in enumerate(df['diagnosis'].unique())}
    torch.backends.cudnn.benchmark = True
    num_workers = min(12, (os.cpu_count() or 4)) if num_workers is None else num_workers
    persistent_workers = num_workers > 0
    if 'median_ita' in df.columns:
        df['skin_tone'] = df['median_ita'].apply(get_skin_tone_label)
        stratify_col = df['skin_tone']
    else:
        stratify_col = None
    if stratify_col is not None:
        _, df_test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True, stratify=stratify_col)
    else:
        _, df_test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    results_root = _get_results_root(output_dir, mode)
    checkpoints_dir = os.path.join(results_root, 'checkpoints')
    fold_indices = []
    if os.path.exists(checkpoints_dir):
        for name in os.listdir(checkpoints_dir):
            if name.startswith('fold_') and name.endswith(f'_{mode}_best.pth'):
                parts = name.split('_')
                if len(parts) >= 4:
                    try:
                        fold_indices.append(int(parts[1]))
                    except ValueError:
                        continue
    num_folds = max(fold_indices) + 1 if fold_indices else 0
    fold_df, per_tone_df = evaluate_test_models(
        df_test,
        label_map,
        output_dir,
        mode,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        num_folds=num_folds,
        image_root=image_root
    )
    if not fold_df.empty:
        print("Testing per fold:")
        print(fold_df)
    if not per_tone_df.empty:
        print("Testing per skin tone:")
        print(per_tone_df)

def make_splits(df, stratify_col, num_folds, seed=42):
    has_groups = 'lesion_id' in df.columns and df['lesion_id'].notna().any()
    if stratify_col is not None and has_groups:
        sg_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(sg_test.split(df, stratify_col, groups=df['lesion_id']))
        dev_idx, test_idx = splits[0]
        df_dev = df.iloc[dev_idx].copy()
        df_test = df.iloc[test_idx].copy()
        kfold = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_generator = kfold.split(df_dev, df_dev['skin_tone'], groups=df_dev['lesion_id'])
        return df_dev, df_test, split_generator
    elif stratify_col is not None:
        df_dev, df_test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True, stratify=stratify_col)
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_generator = kfold.split(df_dev, df_dev['skin_tone'])
        return df_dev, df_test, split_generator
    elif has_groups:
        g_test = GroupKFold(n_splits=5)
        splits = list(g_test.split(df, groups=df['lesion_id']))
        dev_idx, test_idx = splits[0]
        df_dev = df.iloc[dev_idx].copy()
        df_test = df.iloc[test_idx].copy()
        kfold = GroupKFold(n_splits=num_folds)
        split_generator = kfold.split(df_dev, groups=df_dev['lesion_id'])
        return df_dev, df_test, split_generator
    else:
        df_dev, df_test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        split_generator = kfold.split(df_dev)
        return df_dev, df_test, split_generator

def build_dataloader(df, label_map, transform, batch_size, shuffle, num_workers, pin_memory, persistent_workers, image_root=None):
    ds = SkinDs(df, label_map, transform=transform, image_root=image_root)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

def build_model(num_classes, learning_rate):
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to('cuda', memory_format=torch.channels_last)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none')
    use_amp = torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp)
    return model, optimizer, criterion, scaler, use_amp

def compute_weights(df_train_fold, mode, output_dir=None, base_dir=None):
    if mode == 'bs':
        return np.ones(len(df_train_fold))
    Q_ref, distances = calOverallQ(df_train_fold, mode, output_dir=output_dir, base_dir=base_dir)
    weights = calculate_drw_weights(distances)
    return weights

def train_one_epoch(model, train_loader, optimizer, scaler, criterion):
    train_loss_sum = 0.0
    train_samples = 0
    for image, label, weight, _ in train_loader:
        image = image.to('cuda', non_blocking=True, memory_format=torch.channels_last)
        label = label.to('cuda', non_blocking=True)
        weight = weight.to('cuda', non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=scaler.is_enabled()):
            output = model(image)
            loss = (criterion(output, label) * weight).sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss_sum += loss.item()
        train_samples += label.size(0)
    return train_loss_sum / train_samples if train_samples > 0 else 0.0

def evaluate_and_log(model, val_loader, criterion, output_dir, mode, fold_index, epoch, train_loss_avg):
    val_loss, val_accuracy, val_f1_macro, val_f1_weighted, per_group_df, _ = evaluate(model, val_loader, criterion)
    save_evaluation(output_dir, mode, fold_index, epoch, train_loss_avg, val_loss, val_accuracy, val_f1_macro, val_f1_weighted, per_group_df)
    return val_loss, val_accuracy, val_f1_macro, val_f1_weighted, per_group_df

def save_best_checkpoint(output_dir, mode, fold_index, model, best_f1_macro, best_epoch, label_map):
    results_root = _get_results_root(output_dir, mode)
    checkpoint_dir = os.path.join(results_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold_index}_{mode}_best.pth')
    torch.save({
        'state_dict': model.state_dict(),
        'best_f1_macro': best_f1_macro,
        'epoch': best_epoch,
        'fold': fold_index,
        'label_map': label_map
    }, checkpoint_path)

label_map = {}
def training(df=None, csv_path=None, output_dir=None, img_dir=None, mode='fs', epochs=10, batch_size=256, num_folds=7, num_workers=None, pin_memory=True, learning_rate=1e-4, base_dir=None, preprocess_config=None, preprocess_config_path=None, seed=42):
    paths = _resolve_paths(csv_path=csv_path, output_dir=output_dir, img_dir=img_dir, base_dir=base_dir)
    csv_path = paths['csv_path']
    output_dir = paths['output_dir']
    image_root = paths['img_dir']
    df = prepare_data(df, csv_path, output_dir, base_dir=base_dir, config=preprocess_config, config_path=preprocess_config_path)
    label_map = {diag: i for i, diag in enumerate(df['diagnosis'].unique())}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    num_workers = min(12, (os.cpu_count() or 4)) if num_workers is None else num_workers
    persistent_workers = num_workers > 0
    if 'median_ita' in df.columns:
        df['skin_tone'] = df['median_ita'].apply(get_skin_tone_label)
        print("\nSkin Tone Distribution (Full Dataset):")
        print(df['skin_tone'].value_counts().sort_index())
        
        stratify_col = df['skin_tone']
    else:
        print("\nWarning: 'median_ita' column not found. Stratified sampling disabled.")
        stratify_col = None
    os.makedirs(output_dir, exist_ok=True)
    df_dev, df_test, split_generator = make_splits(df, stratify_col, num_folds, seed=seed)
    fold_train_images = []
    fold_Q_refs = []
    best_overall_f1_macro = -1.0
    best_overall_state = None
    best_overall_fold = None
    best_overall_epoch = None
    
    for fold_index, (train_idx, val_idx) in enumerate(split_generator):
        df_train_fold = df_dev.iloc[train_idx].copy()
        df_val_fold = df_dev.iloc[val_idx].copy()
        df_train_fold['weight'] = compute_weights(df_train_fold, mode, output_dir=output_dir, base_dir=base_dir)
        train_loader = build_dataloader(df_train_fold, label_map, train_transform, batch_size, True, num_workers, pin_memory, persistent_workers, image_root=image_root)
        val_loader = build_dataloader(df_val_fold, label_map, val_transform, batch_size, False, num_workers, pin_memory, persistent_workers, image_root=image_root)
        model, optimizer, criterion, scaler, _ = build_model(len(label_map), learning_rate)
        best_fold_f1_macro = -1.0
        best_fold_epoch = None
        for epoch in range(epochs):
            train_loss_avg = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
            val_loss, val_accuracy, val_f1_macro, val_f1_weighted, per_group_df = evaluate_and_log(model, val_loader, criterion, output_dir, mode, fold_index, epoch + 1, train_loss_avg)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss_avg:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
            for _, row in per_group_df.iterrows():
                print(
                    f"  {row['skin_tone']}: Acc = {row['accuracy']:.4f}, "
                    f"F1_w = {row['f1_weighted']:.4f}, F1_m = {row['f1_macro']:.4f}, "
                    f"Support = {int(row['support'])}"
                )
            if val_f1_macro > best_fold_f1_macro:
                best_fold_f1_macro = val_f1_macro
                best_fold_epoch = epoch + 1
                save_best_checkpoint(output_dir, mode, fold_index, model, best_fold_f1_macro, best_fold_epoch, label_map)
            if val_f1_macro > best_overall_f1_macro:
                best_overall_f1_macro = val_f1_macro
                best_overall_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_overall_fold = fold_index
                best_overall_epoch = epoch + 1
        if best_fold_epoch is not None:
            print(f"Fold {fold_index} Best F1 (macro): {best_fold_f1_macro:.4f} at Epoch {best_fold_epoch}")
    if best_overall_state is not None:
        results_root = _get_results_root(output_dir, mode)
        best_model_path = os.path.join(results_root, f'best_model_f1_{mode}.pth')
        torch.save({
            'state_dict': best_overall_state,
            'best_f1_macro': best_overall_f1_macro,
            'epoch': best_overall_epoch,
            'fold': best_overall_fold,
            'label_map': label_map
        }, best_model_path)
        print(f"Best model saved to {best_model_path} (F1 macro: {best_overall_f1_macro:.4f})")
    fold_df, per_tone_df = evaluate_test_models(
        df_test,
        label_map,
        output_dir,
        mode,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        num_folds=num_folds,
        image_root=image_root
    )
    if not fold_df.empty:
        print("Testing per fold:")
        print(fold_df)
    if not per_tone_df.empty:
        print("Testing per skin tone:")
        print(per_tone_df)
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--mode', choices=['fs', 'wd', 'pf', 'bs'], default='fs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-folds', type=int, default=7)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--run-test-only', action='store_true', default=False)
    parser.add_argument('--preprocess-config', default=None)
    parser.add_argument('--pin-memory', dest='pin_memory', action='store_true')
    parser.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')
    parser.set_defaults(pin_memory=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    paths = _resolve_paths(csv_path=args.csv_path, output_dir=args.output_dir)
    csv_path = paths['csv_path']
    output_dir = paths['output_dir']
    img_dir = paths['img_dir']
    mask_dir = paths['mask_dir']
    base_dir = paths['base_dir']
    mode = args.mode
    start_time = time.time()
    run_test_only = args.run_test_only
    seed = args.seed
    preprocess_config_path = args.preprocess_config
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_folds = args.num_folds
    learning_rate = args.learning_rate
    pin_memory = args.pin_memory
    run_params = {
        'csv_path': csv_path,
        'output_dir': output_dir,
        'mode': mode,
        'epochs': epochs,
        'batch_size': batch_size,
        'num_folds': num_folds,
        'num_workers': num_workers,
        'learning_rate': learning_rate,
        'pin_memory': pin_memory,
        'run_test_only': run_test_only,
        'seed': seed
    }
    results_root = _get_results_root(output_dir, mode)
    os.makedirs(results_root, exist_ok=True)
    pd.DataFrame([run_params]).to_csv(os.path.join(results_root, f'run_params_{mode}.csv'), index=False)
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        print("=== 步骤1：运行预处理生成质量特征 ===")
        preprocessing(csv_path=csv_path, output_dir=output_dir, img_dir=img_dir, mask_dir=mask_dir, base_dir=base_dir)
    else:
        print(f"✅ 预处理数据已存在：{len(os.listdir(output_dir))} 个文件")
    if run_test_only:
        print("\n=== 步骤2：开始测试流程（仅测试） ===")
        run_testing_only(csv_path=csv_path, output_dir=output_dir, img_dir=img_dir, mode=mode, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, base_dir=base_dir, preprocess_config_path=preprocess_config_path, seed=seed)
    else:
        print("\n=== 步骤2：开始训练流程 ===")
        training(csv_path=csv_path, output_dir=output_dir, img_dir=img_dir, mode=mode, epochs=epochs, batch_size=batch_size, num_folds=num_folds, num_workers=num_workers, pin_memory=pin_memory, learning_rate=learning_rate, base_dir=base_dir, preprocess_config_path=preprocess_config_path, seed=seed)
    elapsed = time.time() - start_time
    print(f"\n=== 总耗时：{elapsed:.2f} 秒 ===")
