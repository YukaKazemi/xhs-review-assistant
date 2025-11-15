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
import numpy as np # <--- 这里是唯一的、关键的补充！

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================================
# 1. 配置区域 - 这是您唯一需要修改的地方
# ======================================================================================
# --- 输入和输出文件 ---
INPUT_FILE_TO_PREDICT = '待筛选博主.xlsx' 
OUTPUT_FILE = '筛选结果.xlsx'

# --- 模型和Scaler路径 ---
MODEL_PATH = 'blogger_classifier_model.pth'
SCALER_PATH = 'scaler.joblib'

# --- 关键！决策阈值 ---
DECISION_THRESHOLD = 0.35 

# --- 其他配置 (请与 process_data.py 保持一致) ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './huggingface_cache'
BLOGGER_INFO_SHEET_NAME = '博主信息'
NOTES_INFO_SHEET_NAME = '博主笔记'
IMAGE_ROOT_FOLDER = '小红书图片'
TEXT_MODEL_NAME = 'moka-ai/m3e-base'
IMAGE_MODEL_NAME = 'sentence-transformers/clip-ViT-B-32'
NUM_NOTES_TO_PROCESS = 20

# ======================================================================================
# 2. 核心代码区域 - 您通常不需要修改以下内容
# ======================================================================================

# --- 从训练脚本中复制过来的必要组件 ---
# class BloggerClassifier(nn.Module):
# train_model.py (优化版)
# ... (您文件顶部的 import 和数据加载部分保持不变)

# 找到您定义 BloggerClassifier 的地方，替换成这个新版本
class BloggerClassifier(nn.Module):
    def __init__(self, input_features):
        super(BloggerClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            # 核心修改：将 Dropout 比例从 0.4 提升到 0.5，进行更强的正则化
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # 同样提升
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # 同样提升

            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# ... (文件剩余的训练循环等部分保持不变)

#     def __init__(self, input_features):
#         super(BloggerClassifier, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_features, 512), nn.ReLU(), nn.Dropout(0.4),
#             nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
#             nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
#             nn.Linear(128, 1), nn.Sigmoid()
#         )
#     def forward(self, x):
#         return self.network(x)

def feature_engineering(df_bloggers, df_notes):
    print("  - Step A: 计算基础特征...")
    final_features_list = []
    notes_grouped = df_notes.groupby('小红书号')
    for _, blogger in tqdm(df_bloggers.iterrows(), total=len(df_bloggers), desc="    - 计算中"):
        blogger_id = str(blogger['小红书号'])
        features = {'小红书号': blogger_id, '用户名': blogger.get('用户名', ''), '个人简介_原始': blogger.get('个人简介', '')}
        bio = str(blogger.get('个人简介', ''))
        features['bio_text'] = bio
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails_found = re.findall(email_pattern, bio)
        features['has_email'] = 1 if emails_found else 0
        followers = blogger.get('粉丝数', 0)
        total_likes = blogger.get('总点赞', 0)
        features['followers'] = followers
        features['total_likes'] = total_likes
        features['avg_likes_per_fan'] = total_likes / (followers + 1)
        if blogger_id in notes_grouped.groups:
            blogger_notes = notes_grouped.get_group(blogger_id).head(NUM_NOTES_TO_PROCESS)
            note_count = len(blogger_notes)
            if note_count > 0:
                likes_list = blogger_notes['笔记点赞数'].tolist()
                single_digit = sum(1 for like in likes_list if like < 10)
                double_digit = sum(1 for like in likes_list if 10 <= like < 100)
                features['single_digit_likes_ratio'] = single_digit / note_count
                features['double_digit_likes_ratio'] = double_digit / note_count
                features['note_titles'] = blogger_notes['笔记标题'].astype(str).tolist()
                features['note_image_paths'] = blogger_notes['笔记封面路径'].astype(str).tolist()
            else:
                features.update({'single_digit_likes_ratio': 0, 'double_digit_likes_ratio': 0, 'note_titles': [], 'note_image_paths': []})
        else:
            features.update({'single_digit_likes_ratio': 0, 'double_digit_likes_ratio': 0, 'note_titles': [], 'note_image_paths': []})
        final_features_list.append(features)
    return pd.DataFrame(final_features_list)

def embed_features(df_features, device):
    print("  - Step B: 转换文本和图片为向量...")
    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
    image_model = SentenceTransformer(IMAGE_MODEL_NAME, device=device)
    bio_texts = df_features['bio_text'].tolist()
    bio_embeddings = text_model.encode(bio_texts, show_progress_bar=True, convert_to_tensor=True, device=device)
    title_avg_embeddings, image_avg_embeddings = [], []
    for _, row in tqdm(df_features.iterrows(), total=len(df_features), desc="    - 转换中"):
        titles = row['note_titles']
        if titles:
            title_embs = text_model.encode(titles, show_progress_bar=False, convert_to_tensor=True, device=device)
            title_avg_embeddings.append(title_embs.mean(axis=0).cpu().numpy())
        else:
            title_avg_embeddings.append([0] * text_model.get_sentence_embedding_dimension())
        image_paths = row['note_image_paths']
        valid_images = []
        if image_paths:
            for img_path in image_paths:
                full_path = os.path.join(IMAGE_ROOT_FOLDER, str(row['小红书号']), img_path)
                if os.path.exists(full_path):
                    try:
                        valid_images.append(Image.open(full_path).convert("RGB"))
                    except Exception:
                        pass
        if valid_images:
            image_embs = image_model.encode(valid_images, show_progress_bar=False, convert_to_tensor=True, device=device)
            image_avg_embeddings.append(image_embs.mean(axis=0).cpu().numpy())
        else:
            image_avg_embeddings.append([0] * 512)
    bio_df = pd.DataFrame(bio_embeddings.cpu().numpy(), columns=[f'bio_vec_{i}' for i in range(bio_embeddings.shape[1])])
    title_df = pd.DataFrame(title_avg_embeddings, columns=[f'title_vec_{i}' for i in range(len(title_avg_embeddings[0]))])
    image_df = pd.DataFrame(image_avg_embeddings, columns=[f'image_vec_{i}' for i in range(len(image_avg_embeddings[0]))])
    df_features = df_features.drop(columns=['bio_text', 'note_titles', 'note_image_paths'])
    final_df = pd.concat([df_features.reset_index(drop=True), bio_df, title_df, image_df], axis=1)
    return final_df

def predict_new_data():
    print("开始执行自动化筛选流程...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到可用设备: {device.upper()}")

    print("Step 1/4: 加载待筛选数据...")
    df_new_bloggers = pd.read_excel(INPUT_FILE_TO_PREDICT, sheet_name=BLOGGER_INFO_SHEET_NAME)
    df_new_notes = pd.read_excel(INPUT_FILE_TO_PREDICT, sheet_name=NOTES_INFO_SHEET_NAME)
    df_new_bloggers['小红书号'] = df_new_bloggers['小红书号'].astype(str)
    df_new_notes['小红书号'] = df_new_notes['小红书号'].astype(str)

    print("Step 2/4: 正在进行特征提取 (此过程较长)...")
    df_features = feature_engineering(df_new_bloggers, df_new_notes)
    df_embedded = embed_features(df_features, device)

    print("Step 3/4: 加载已训练好的模型和数据scaler...")
    scaler = joblib.load(SCALER_PATH)
    model_state_dict = torch.load(MODEL_PATH)
    input_dim = model_state_dict['network.0.weight'].shape[1]
    model = BloggerClassifier(input_features=input_dim)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    print("Step 4/4: 正在执行预测...")
    info_cols = ['小红书号', '用户名', '个人简介_原始']
    feature_cols = [col for col in df_embedded.columns if col not in info_cols]
    X_predict = df_embedded[feature_cols].copy()
    X_predict_scaled = scaler.transform(X_predict)
    X_predict_tensor = torch.tensor(X_predict_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        probabilities = model(X_predict_tensor).cpu().numpy().flatten()

    results_df = df_embedded[info_cols].copy()
    results_df['符合概率'] = probabilities
    results_df['筛选建议'] = np.where(results_df['符合概率'] >= DECISION_THRESHOLD, '建议符合', '建议拒绝')

    results_df = results_df.sort_values(by='符合概率', ascending=False)
    
    results_df.to_excel(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print("🎉 筛选完成！结果已保存。 🎉")
    print(f"请在项目文件夹中查看 '{OUTPUT_FILE}' 文件。")
    print(f"当前使用的决策阈值为: {DECISION_THRESHOLD} (得分高于此值则建议符合)")
    print("您可以随时修改脚本顶部的DECISION_THRESHOLD值来调整筛选的宽松度。")
    print("="*40)

if __name__ == "__main__":
    predict_new_data()

