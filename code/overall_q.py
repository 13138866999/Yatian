from typing import Any
import argparse
import time
from preprocessing import process_image
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

def preprocessing(img_dir = '/root/skinai/data/images', mask_dir = '/root/skinai/data/masks', 
                  output_dir = '/root/skinai/data/qi', dataf = None,
                  csv_path = '/root/skinai/data/GroundTruth.csv'):
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
def prepare_data(df=None, csv_path='/root/skinai/data/GroundTruth.csv', output_dir='/root/skinai/data/qi'):
    if df is None:
        df = pd.read_csv(csv_path)
    disease_columns = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    
    # 检查是否所有病种列都存在
    existing_disease_cols = [col for col in disease_columns if col in df.columns]
    if len(existing_disease_cols) > 0:
        # 获取每个样本的病种（one-hot中为1的列）
        def get_diagnosis(row):
            for col in existing_disease_cols:
                if row[col] == 1.0:
                    return col
            return 'Unknown'  # 如果没有找到1，返回Unknown
        
        df['diagnosis'] = df.apply(get_diagnosis, axis=1)
        
        print(f"\nDisease Distribution (Full Dataset):")
        print(df['diagnosis'].value_counts())
        
        # 删除原来的one-hot列来节省空间
        df = df.drop(columns=existing_disease_cols)
        print(f"Removed one-hot columns: {existing_disease_cols}")
    else:
        if 'diagnosis' in df.columns:
            df['diagnosis'] = df['diagnosis'].astype(str)
        elif 'dx' in df.columns:
            df['diagnosis'] = df['dx'].astype(str)
        else:
            df['diagnosis'] = 'Unknown'
        print("\nDisease Distribution (Full Dataset):")
        print(df['diagnosis'].value_counts())
        print("\nWarning: No disease columns found for diagnosis conversion.")
    # --- Data Merging Logic ---
    ita_csv_path = os.path.join(output_dir, 'ita_medians.csv')
    if os.path.exists(ita_csv_path):
        ita_df = pd.read_csv(ita_csv_path)
        
        # Drop NaNs
        ita_df = ita_df.dropna(subset=['median_ita'])
        
        # Inner Join
        df = pd.merge(df, ita_df, on='image', how='inner')
        print(f"Merged data shape: {df.shape}")
        if not df.empty:
            print("First row sample:")
            print(df.iloc[0])
        else:
            print(f"Warning: {ita_csv_path} not found. Using original GroundTruth only.")

    # 关联 HAM10000 元数据，获取 lesion_id
    meta_path = '/root/skinai/data/HAM10000_metadata.csv'
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path, usecols=['lesion_id', 'image_id'])
        df = pd.merge(df, meta, left_on='image', right_on='image_id', how='left')
        df = df.drop(columns=['image_id'])
        print("Joined HAM10000 metadata: added lesion_id")
    else:
        raise FileNotFoundError(f"CRITICAL: {meta_path} missing! Cannot perform lesion-level split.")
    
    return df# --------------------------

    # Generate Skin Tone Labels for Stratification
    
    
    # Convert one-hot encoded disease columns to single diagnosis column
    
    return df

def calOverallQ(df, mode = 'bs'):
    methods = ['fs', 'wd', 'pf', 'bs']
    if mode not in methods:
        raise ValueError(f"Mode {mode} not in {methods}")
    
    train_histograms = []
    missing_files = []
    
    for img_id in df['image']:
        file_path = f'/root/skinai/data/qi/{img_id}.npy'
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

def evaluate_test_models(df_test, label_map, output_dir, mode, batch_size, num_workers, pin_memory, persistent_workers, num_folds):
    results_root = _get_results_root(output_dir, mode)
    results_dir = os.path.join(results_root, 'testing')
    os.makedirs(results_dir, exist_ok=True)
    skin_ds_test = SkinDs(df_test, label_map, transform=val_transform)
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

def run_testing_only(csv_path='/root/skinai/data/GroundTruth.csv', output_dir='/root/skinai/data/qi', mode='fs', batch_size=256, num_workers=None, pin_memory=True):
    df = prepare_data(None, csv_path, output_dir)
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
        _, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=stratify_col)
    else:
        _, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
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
        num_folds=num_folds
    )
    if not fold_df.empty:
        print("Testing per fold:")
        print(fold_df)
    if not per_tone_df.empty:
        print("Testing per skin tone:")
        print(per_tone_df)

label_map = {}
def training(df=None, csv_path='/root/skinai/data/GroundTruth.csv', output_dir='/root/skinai/data/qi', mode='fs', epochs=10, batch_size=256, num_folds=7, num_workers=None, pin_memory=True, learning_rate=1e-4):
    df = prepare_data(df, csv_path, output_dir)
    label_map = {diag: i for i, diag in enumerate(df['diagnosis'].unique())}
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
    has_groups = 'lesion_id' in df.columns and df['lesion_id'].notna().any()
    if stratify_col is not None and has_groups:
        sg_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(sg_test.split(df, stratify_col, groups=df['lesion_id']))
        dev_idx, test_idx = splits[0]
        df_dev = df.iloc[dev_idx].copy()
        df_test = df.iloc[test_idx].copy()
        kfold = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=42)
        split_generator = kfold.split(df_dev, df_dev['skin_tone'], groups=df_dev['lesion_id'])
    elif stratify_col is not None:
        df_dev, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=stratify_col)
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        split_generator = kfold.split(df_dev, df_dev['skin_tone'])
    elif has_groups:
        # 无肤色分层时，按 lesion 分组保持同病灶不跨集
        g_test = GroupKFold(n_splits=5)
        splits = list(g_test.split(df, groups=df['lesion_id']))
        dev_idx, test_idx = splits[0]
        df_dev = df.iloc[dev_idx].copy()
        df_test = df.iloc[test_idx].copy()
        kfold = GroupKFold(n_splits=num_folds)
        split_generator = kfold.split(df_dev, groups=df_dev['lesion_id'])
    else:
        df_dev, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        split_generator = kfold.split(df_dev)
    fold_train_images = []
    fold_Q_refs = []
    best_overall_f1_macro = -1.0
    best_overall_state = None
    best_overall_fold = None
    best_overall_epoch = None
    
    for fold_index, (train_idx, val_idx) in enumerate(split_generator):
        df_train_fold = df_dev.iloc[train_idx].copy()
        df_val_fold = df_dev.iloc[val_idx].copy()
        if mode == 'bs':
            df_train_fold['weight'] = 1.0
        else:
            Q_ref, distances = calOverallQ(df_train_fold, mode)
            weights = calculate_drw_weights(distances)
            df_train_fold['weight'] = weights
        skin_ds_train = SkinDs(df_train_fold, label_map, transform=train_transform)
        train_loader = DataLoader(
            skin_ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        skin_ds_val = SkinDs(df_val_fold, label_map, transform=val_transform)
        val_loader = DataLoader(
            skin_ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

        model = models.resnet50(weights='IMAGENET1K_V1')        
        model.fc = nn.Linear(model.fc.in_features, len(label_map))
        model = model.to('cuda', memory_format=torch.channels_last)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        use_amp = torch.cuda.is_available()
        scaler = GradScaler('cuda', enabled=use_amp)
        best_fold_f1_macro = -1.0
        best_fold_epoch = None
        for epoch in range(epochs):
            train_loss_sum = 0.0
            train_samples = 0
            for image, label, weight, _ in train_loader:
                image = image.to('cuda', non_blocking=True, memory_format=torch.channels_last)
                label = label.to('cuda', non_blocking=True)
                weight = weight.to('cuda', non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', enabled=use_amp):
                    output = model(image)
                    loss = (criterion(output, label) * weight).sum()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_sum += loss.item()
                train_samples += label.size(0)
            train_loss_avg = train_loss_sum / train_samples if train_samples > 0 else 0.0
            val_loss, val_accuracy, val_f1_macro, val_f1_weighted, per_group_df, _ = evaluate(model, val_loader, criterion)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss_avg:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
            for _, row in per_group_df.iterrows():
                print(
                    f"  {row['skin_tone']}: Acc = {row['accuracy']:.4f}, "
                    f"F1_w = {row['f1_weighted']:.4f}, F1_m = {row['f1_macro']:.4f}, "
                    f"Support = {int(row['support'])}"
                )
            save_evaluation(output_dir, mode, fold_index, epoch + 1, train_loss_avg, val_loss, val_accuracy, val_f1_macro, val_f1_weighted, per_group_df)
            if val_f1_macro > best_fold_f1_macro:
                best_fold_f1_macro = val_f1_macro
                best_fold_epoch = epoch + 1
                results_root = _get_results_root(output_dir, mode)
                checkpoint_dir = os.path.join(results_root, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold_index}_{mode}_best.pth')
                torch.save({
                    'state_dict': model.state_dict(),
                    'best_f1_macro': best_fold_f1_macro,
                    'epoch': best_fold_epoch,
                    'fold': fold_index,
                    'label_map': label_map
                }, checkpoint_path)
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
        num_folds=num_folds
    )
    if not fold_df.empty:
        print("Testing per fold:")
        print(fold_df)
    if not per_tone_df.empty:
        print("Testing per skin tone:")
        print(per_tone_df)
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', default='/root/skinai/data/GroundTruth.csv')
    parser.add_argument('--output-dir', default='/root/skinai/data/qi')
    parser.add_argument('--mode', choices=['fs', 'wd', 'pf', 'bs'], default='fs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-folds', type=int, default=7)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--run-test-only', action='store_true', default=False)
    parser.add_argument('--pin-memory', dest='pin_memory', action='store_true')
    parser.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')
    parser.set_defaults(pin_memory=True)
    args = parser.parse_args()

    csv_path = args.csv_path
    output_dir = args.output_dir
    mode = args.mode
    start_time = time.time()
    run_test_only = args.run_test_only
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
        'run_test_only': run_test_only
    }
    results_root = _get_results_root(output_dir, mode)
    os.makedirs(results_root, exist_ok=True)
    pd.DataFrame([run_params]).to_csv(os.path.join(results_root, f'run_params_{mode}.csv'), index=False)
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        print("=== 步骤1：运行预处理生成质量特征 ===")
        preprocessing(csv_path=csv_path, output_dir=output_dir)
    else:
        print(f"✅ 预处理数据已存在：{len(os.listdir(output_dir))} 个文件")
    if run_test_only:
        print("\n=== 步骤2：开始测试流程（仅测试） ===")
        run_testing_only(csv_path=csv_path, output_dir=output_dir, mode=mode, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    else:
        print("\n=== 步骤2：开始训练流程 ===")
        training(csv_path=csv_path, output_dir=output_dir, mode=mode, epochs=epochs, batch_size=batch_size, num_folds=num_folds, num_workers=num_workers, pin_memory=pin_memory, learning_rate=learning_rate)
    elapsed = time.time() - start_time
    print(f"\n=== 总耗时：{elapsed:.2f} 秒 ===")
