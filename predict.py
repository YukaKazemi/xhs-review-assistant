# predict.py (最终版 v3.0 - 带人工审核区)
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

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================================
# 1. 配置区域 - 您现在可以控制两个阈值！
# ======================================================================================
# --- 输入和输出文件 ---
INPUT_FILE_TO_PREDICT = '全部测试数据.xlsx' 
OUTPUT_FILE = '筛选结果(带人工审核).xlsx' # 最终输出文件

# --- 模型和Scaler路径 ---
MODEL_PATH = 'blogger_classifier_model.pth'
SCALER_PATH = 'scaler.joblib'

# --- 关键！三段式决策阈值 ---
UPPER_THRESHOLD = 0.6  # 高于或等于此值，直接判定为“自动符合”
LOWER_THRESHOLD = 0.4  # 低于或等于此值，直接判定为“自动拒绝”
# 介于 UPPER 和 LOWER 之间的，将被判定为“待人工审核”

# --- 其他配置 (保持不变) ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './huggingface_cache'
BLOGGER_INFO_SHEET_NAME = '博主信息'
NOTES_INFO_SHEET_NAME = '博主笔记'
IMAGE_ROOT_FOLDER = '小红书图片'
TEXT_MODEL_NAME = 'moka-ai/m3e-base'
IMAGE_MODEL_NAME = 'sentence-transformers/clip-ViT-B-32'
NUM_NOTES_TO_PROCESS = 20

# ======================================================================================
# 2. 核心代码区域 - 只有一处小修改
# ======================================================================================
# (BloggerClassifier, feature_engineering, embed_features 函数都保持不变，这里省略以节省空间)
# (您可以直接在您现有的 predict.py 上修改，无需复制这些函数)
class BloggerClassifier(nn.Module):
    def __init__(self, input_features):
        super(BloggerClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

def feature_engineering(df_bloggers, df_notes):
    print("  - Step A: 计算基础特征...")
    final_features_list = []
    notes_grouped = df_notes.groupby('小红书号')
    for _, blogger in tqdm(df_bloggers.iterrows(), total=len(df_bloggers), desc="    - 计算中"):
        blogger_id = str(blogger['小红书号'])
        features = {
            '小红书号': blogger_id, 
            '用户名': blogger.get('用户名', ''), 
            '个人简介_原始': blogger.get('个人简介', ''),
            '原始审核状态': blogger.get('原始审核状态', '未知')
        }
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
    info_cols = ['小红书号', '用户名', '个人简介_原始', '原始审核状态']
    feature_cols = [col for col in df_embedded.columns if col not in info_cols]
    
    X_predict = df_embedded[feature_cols].copy()
    X_predict_scaled = scaler.transform(X_predict)
    X_predict_tensor = torch.tensor(X_predict_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        probabilities = model(X_predict_tensor).cpu().numpy().flatten()

    results_df = df_embedded[info_cols].copy()
    results_df['符合概率'] = probabilities
    
    # === 唯一的、核心的修改在这里！ ===
    # 定义三段式逻辑的条件和选择
    conditions = [
        results_df['符合概率'] >= UPPER_THRESHOLD,
        results_df['符合概率'] <= LOWER_THRESHOLD
    ]
    choices = ['自动符合', '自动拒绝']
    
    # 使用 np.select 来应用三段式逻辑
    results_df['筛选建议'] = np.select(conditions, choices, default='待人工审核')
    
    results_df = results_df.sort_values(by='符合概率', ascending=False)
    
    final_cols = ['小红书号', '用户名', '原始审核状态', '符合概率', '筛选建议', '个人简介_原始']
    results_df = results_df[final_cols]
    
    results_df.to_excel(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print("🎉 筛选完成！结果已保存。 🎉")
    print(f"请在项目文件夹中查看 '{OUTPUT_FILE}' 文件。")
    print(f"当前使用的决策逻辑是: >= {UPPER_THRESHOLD} -> 自动符合, <= {LOWER_THRESHOLD} -> 自动拒绝, 中间 -> 待人工审核")
    print("="*40)

if __name__ == "__main__":
    predict_new_data()
