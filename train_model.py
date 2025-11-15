# train_model.py (v3.0 - 最终自洽版，自动创建并保存Scaler)
# -*- coding: utf-8 -*-
import pandas as pd
import torch
import joblib
import os
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch.nn as nn
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================================
# 1. 配置区域
# ======================================================================================
# --- 输入和输出文件 ---
INPUT_FILE_WITH_LABELS = '待训练数据.xlsx' 
MODEL_PATH = 'blogger_classifier_model.pth'
SCALER_PATH = 'scaler.joblib'

# --- 核心调优参数 ---
VALIDATION_SET_SIZE = 0.15
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 7

# --- 其他配置 (与您项目保持一致) ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './huggingface_cache'
BLOGGER_INFO_SHEET_NAME = '博主信息'
NOTES_INFO_SHEET_NAME = '博主笔记'
IMAGE_ROOT_FOLDER = '小红书图片'
TEXT_MODEL_NAME = 'moka-ai/m3e-base'
IMAGE_MODEL_NAME = 'sentence-transformers/clip-ViT-B-32'
NUM_NOTES_TO_PROCESS = 20

# ======================================================================================
# 2. 模型定义 (与之前一致的优化版)
# ======================================================================================
class BloggerClassifier(nn.Module):
    def __init__(self, input_features):
        super(BloggerClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

# ======================================================================================
# 3. 特征工程函数 (与您 predict.py 中的函数完全一致)
# ======================================================================================
def feature_engineering(df_bloggers, df_notes):
    print("  - Step A: 计算基础特征...")
    final_features_list = []
    notes_grouped = df_notes.groupby('小红书号')
    for _, blogger in tqdm(df_bloggers.iterrows(), total=len(df_bloggers), desc="    - 计算中"):
        blogger_id = str(blogger['小红书号'])
        features = {'小红书号': blogger_id, '用户名': blogger.get('用户名', ''), '个人简介_原始': blogger.get('个人简介', ''), '审核状态': blogger.get('审核状态', '未知')}
        bio = str(blogger.get('个人简介', ''))
        features['bio_text'] = bio
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails_found = re.findall(email_pattern, bio)
        features['has_email'] = 1 if emails_found else 0
        followers = blogger.get('粉丝数', 0)
        total_likes = blogger.get('总点赞', 0)
        features['followers'] = followers
        features['total_likes'] = total_likes
        features['avg_likes_per_fan'] = total_likes / (followers + 1) if followers > 0 else total_likes
        if blogger_id in notes_grouped.groups:
            blogger_notes = notes_grouped.get_group(blogger_id).head(NUM_NOTES_TO_PROCESS)
            note_count = len(blogger_notes)
            if note_count > 0:
                likes_list = blogger_notes['笔记点赞数'].tolist()
                single_digit = sum(1 for like in likes_list if like < 10); double_digit = sum(1 for like in likes_list if 10 <= like < 100)
                features['single_digit_likes_ratio'] = single_digit / note_count
                features['double_digit_likes_ratio'] = double_digit / note_count
                features['note_titles'] = blogger_notes['笔记标题'].astype(str).tolist()
                features['note_image_paths'] = blogger_notes['笔记封面路径'].astype(str).tolist()
            else: features.update({'single_digit_likes_ratio': 0,'double_digit_likes_ratio': 0,'note_titles': [],'note_image_paths': []})
        else: features.update({'single_digit_likes_ratio': 0,'double_digit_likes_ratio': 0,'note_titles': [],'note_image_paths': []})
        final_features_list.append(features)
    return pd.DataFrame(final_features_list)

def embed_features(df_features, device):
    print("  - Step B: 转换文本和图片为向量...")
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
    image_model = SentenceTransformer(IMAGE_MODEL_NAME, device=device)
    bio_embeddings = text_model.encode(df_features['bio_text'].tolist(), show_progress_bar=True, convert_to_tensor=True, device=device)
    title_avg_embeddings, image_avg_embeddings = [], []
    for _, row in tqdm(df_features.iterrows(), total=len(df_features), desc="    - 向量化"):
        if row['note_titles']: title_avg_embeddings.append(text_model.encode(row['note_titles'], show_progress_bar=False, convert_to_tensor=True, device=device).mean(axis=0).cpu().numpy())
        else: title_avg_embeddings.append([0] * text_model.get_sentence_embedding_dimension())
        valid_images = []
        if row['note_image_paths']:
            for img_path in row['note_image_paths']:
                full_path = os.path.join(IMAGE_ROOT_FOLDER, str(row['小红书号']), img_path)
                if os.path.exists(full_path):
                    try: valid_images.append(Image.open(full_path).convert("RGB"))
                    except: pass
        if valid_images: image_avg_embeddings.append(image_model.encode(valid_images, show_progress_bar=False, convert_to_tensor=True, device=device).mean(axis=0).cpu().numpy())
        else: image_avg_embeddings.append([0] * 512)
    bio_df = pd.DataFrame(bio_embeddings.cpu().numpy(), columns=[f'bio_vec_{i}' for i in range(bio_embeddings.shape[1])])
    title_df = pd.DataFrame(title_avg_embeddings, columns=[f'title_vec_{i}' for i in range(len(title_avg_embeddings[0]))])
    image_df = pd.DataFrame(image_avg_embeddings, columns=[f'image_vec_{i}' for i in range(len(image_avg_embeddings[0]))])
    return pd.concat([df_features.drop(columns=['bio_text', 'note_titles', 'note_image_paths']).reset_index(drop=True), bio_df, title_df, image_df], axis=1)

# ======================================================================================
# 4. 核心训练流程 - 全新自洽版
# ======================================================================================
def train_model_from_scratch():
    print("开始执行模型训练流程 (v3.0 - 完全自洽版)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到可用设备: {device.upper()}")

    # --- Step 1: 加载原始Excel数据 ---
    print("Step 1/5: 加载原始Excel数据...")
    if not os.path.exists(INPUT_FILE_WITH_LABELS):
        print(f"错误: 找不到输入文件 '{INPUT_FILE_WITH_LABELS}'。请确保文件存在。")
        return
    df_bloggers = pd.read_excel(INPUT_FILE_WITH_LABELS, sheet_name=BLOGGER_INFO_SHEET_NAME)
    df_notes = pd.read_excel(INPUT_FILE_WITH_LABELS, sheet_name=NOTES_INFO_SHEET_NAME)
    df_bloggers['小红书号'] = df_bloggers['小红书号'].astype(str)
    df_notes['小红书号'] = df_notes['小红书号'].astype(str)

    # --- Step 2: 特征工程与向量化 ---
    print("Step 2/5: 执行特征工程与向量化...")
    df_features = feature_engineering(df_bloggers, df_notes)
    df_embedded = embed_features(df_features, device)
    
    df_embedded = df_embedded.dropna(subset=['审核状态'])
    df_embedded['label'] = df_embedded['审核状态'].apply(lambda x: 1 if x == '符合' else 0)

    info_cols = ['小红书号', '用户名', '个人简介_原始', '审核状态', 'label']
    feature_cols = [col for col in df_embedded.columns if col not in info_cols]
    
    X = df_embedded[feature_cols]
    y = df_embedded['label'].values

    # --- Step 3: 划分数据, 创建并保存Scaler ---
    print("Step 3/5: 划分数据集并创建/保存Scaler...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SET_SIZE, random_state=42, stratify=y
    )

    # <--- 核心修改：在这里创建、拟合并保存新的Scaler --->
    print(f"  - 正在使用 {len(X_train)} 条训练数据拟合新的Scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # 只在训练数据上fit_transform
    X_val_scaled = scaler.transform(X_val)       # 在验证数据上只transform

    joblib.dump(scaler, SCALER_PATH)
    print(f"  - 全新的Scaler已保存至 '{SCALER_PATH}'")
    
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).view(-1, 1))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Step 4: 初始化模型、损失函数和优化器 ---
    print("Step 4/5: 初始化模型和优化器...")
    input_dim = X_train_scaled.shape[1]
    model = BloggerClassifier(input_features=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # --- Step 5: 执行训练和验证循环 (带早停) ---
    print("Step 5/5: 开始训练...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(features); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features); loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}")
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Validation loss improved. Saving new best model to '{MODEL_PATH}'")
        else:
            epochs_no_improve += 1
            print(f"  -> Validation loss did not improve for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break
            
    print("\n" + "="*40)
    print("🎉 模型训练完成！ 🎉")
    print(f"最优模型已保存到: '{MODEL_PATH}'")
    print(f"配套的Scaler已保存到: '{SCALER_PATH}'")
    print("现在您可以安全地使用 predict.py 脚本进行预测了。")
    print("="*40)


if __name__ == "__main__":
    train_model_from_scratch()
