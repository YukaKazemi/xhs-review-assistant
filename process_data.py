# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import re
from tqdm import tqdm

# ======================================================================================
# 1. 配置区域 - 请根据您的实际情况修改这里的路径和文件名
# ======================================================================================
# 设置环境变量，使用国内镜像源加速模型下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './huggingface_cache' # 将模型缓存到当前项目文件夹下
# --- 文件路径配置 ---
POSITIVE_FILE = '已通过数据.xlsx'
NEGATIVE_FILE = '未通过数据.xlsx'

# --- 博主信息和笔记信息在Excel中的工作表名称 ---
# 根据您的文件，博主信息在'博主信息' sheet，笔记在'博主笔记' sheet
# 如果您的sheet名称不同，请在这里修改
BLOGGER_INFO_SHEET_NAME = '博主信息'
NOTES_INFO_SHEET_NAME = '博主笔记' # 假设笔记数据在名为'博主笔记'的sheet里

# --- 图片根目录 ---
# 这是您存放所有博主图片的大文件夹的名称
IMAGE_ROOT_FOLDER = '小红书图片'

# --- 输出文件 ---
# 这是程序运行后生成的最终训练数据文件名
OUTPUT_CSV_FILE = 'final_training_data.csv'

# --- 模型配置 ---
# 我们使用这个模型来将文本（简介、标题）转换为向量
# 首次运行时会自动从网上下载，请保持网络连接
TEXT_MODEL_NAME = 'moka-ai/m3e-base' 
# 我们使用CLIP模型来将图片转换为向量
IMAGE_MODEL_NAME = 'sentence-transformers/clip-ViT-B-32'

# --- 其他配置 ---
# 每个博主分析的笔记数量
NUM_NOTES_TO_PROCESS = 20

# ======================================================================================
# 2. 核心代码区域 - 您通常不需要修改以下内容
# ======================================================================================

def load_and_prepare_data():
    """加载'符合'和'不符合'的数据，并打上标签"""
    print("Step 1/5: 正在加载并合并博主数据...")
    
    # 加载博主信息
    df_pos_blogger = pd.read_excel(POSITIVE_FILE, sheet_name=BLOGGER_INFO_SHEET_NAME)
    df_pos_blogger['label'] = 1
    
    df_neg_blogger = pd.read_excel(NEGATIVE_FILE, sheet_name=BLOGGER_INFO_SHEET_NAME)
    df_neg_blogger['label'] = 0
    
    df_bloggers = pd.concat([df_pos_blogger, df_neg_blogger], ignore_index=True)
    print(f"  - 成功合并 {len(df_bloggers)} 条博主信息。")

    # 加载笔记信息
    print("  - 正在加载笔记数据...")
    df_pos_notes = pd.read_excel(POSITIVE_FILE, sheet_name=NOTES_INFO_SHEET_NAME)
    df_neg_notes = pd.read_excel(NEGATIVE_FILE, sheet_name=NOTES_INFO_SHEET_NAME)
    df_notes = pd.concat([df_pos_notes, df_neg_notes], ignore_index=True)
    # 确保'小红书号'类型一致以便合并
    df_bloggers['小红书号'] = df_bloggers['小红书号'].astype(str)
    df_notes['小红书号'] = df_notes['小红书号'].astype(str)
    
    print(f"  - 成功加载 {len(df_notes)} 条笔记信息。")
    return df_bloggers, df_notes

def feature_engineering(df_bloggers, df_notes):
    """为每个博主计算数值特征和文本特征"""
    print("Step 2/5: 开始进行特征工程...")

    final_features_list = []
    
    # 将笔记按博主分组，方便快速查找
    notes_grouped = df_notes.groupby('小红书号')

    # 使用tqdm显示进度条
    for _, blogger in tqdm(df_bloggers.iterrows(), total=len(df_bloggers), desc="  - 处理博主中"):
        blogger_id = str(blogger['小红书号'])
        features = {'小红书号': blogger_id, 'label': blogger['label']}

        # --- 特征1: 个人简介处理 ---
        bio = str(blogger.get('个人简介', ''))
        features['bio_text'] = bio # 保存原始文本，后续统一编码

        # --- 特征2: 邮箱识别 ---
        # 强大的正则表达式，可以识别各种被混淆的邮箱格式
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails_found = re.findall(email_pattern, bio)
        features['has_email'] = 1 if emails_found else 0

        # --- 特征3: 基础数值特征 ---
        followers = blogger.get('粉丝数', 0)
        total_likes = blogger.get('总点赞', 0)
        features['followers'] = followers
        features['total_likes'] = total_likes
        features['avg_likes_per_fan'] = total_likes / (followers + 1) # +1防止除以0

        # --- 特征4: 笔记相关特征 ---
        if blogger_id in notes_grouped.groups:
            blogger_notes = notes_grouped.get_group(blogger_id).head(NUM_NOTES_TO_PROCESS)
            note_count = len(blogger_notes)

            if note_count > 0:
                likes_list = blogger_notes['笔记点赞数'].tolist()
                
                # 计算点赞数分布
                single_digit = sum(1 for like in likes_list if like < 10)
                double_digit = sum(1 for like in likes_list if 10 <= like < 100)
                
                features['single_digit_likes_ratio'] = single_digit / note_count
                features['double_digit_likes_ratio'] = double_digit / note_count
                
                # 保存笔记标题和封面路径，后续统一编码
                features['note_titles'] = blogger_notes['笔记标题'].astype(str).tolist()
                # 假设封面路径在'笔记封面路'列中
                features['note_image_paths'] = blogger_notes['笔记封面路径'].astype(str).tolist()
            else:
                # 如果没有笔记，则赋予默认值
                features['single_digit_likes_ratio'] = 0
                features['double_digit_likes_ratio'] = 0
                features['note_titles'] = []
                features['note_image_paths'] = []
        else:
            # 如果在笔记数据中找不到该博主，同样赋予默认值
            features['single_digit_likes_ratio'] = 0
            features['double_digit_likes_ratio'] = 0
            features['note_titles'] = []
            features['note_image_paths'] = []
        
        final_features_list.append(features)

    return pd.DataFrame(final_features_list)

def embed_features(df_features):
    """使用模型将文本和图片转换为向量"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Step 3/5: 加载文本和图像模型到 {device}...")

    text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
    image_model = SentenceTransformer(IMAGE_MODEL_NAME, device=device)
    
    print("Step 4/5: 开始将文本和图片转换为向量（此过程可能需要较长时间）...")
    
    bio_texts = df_features['bio_text'].tolist()
    bio_embeddings = text_model.encode(bio_texts, show_progress_bar=True, convert_to_tensor=True, device=device)
    
    title_avg_embeddings = []
    image_avg_embeddings = []

    for _, row in tqdm(df_features.iterrows(), total=len(df_features), desc="  - 处理笔记序列"):
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
                blogger_folder_name = row['小红书号']
                full_path = os.path.join(IMAGE_ROOT_FOLDER, blogger_folder_name, img_path)
                
                if os.path.exists(full_path):
                    try:
                        valid_images.append(Image.open(full_path).convert("RGB"))
                    except Exception as e:
                        print(f"    - 警告: 无法打开图片 {full_path}, 已跳过. 错误: {e}")
        
        if valid_images:
            image_embs = image_model.encode(valid_images, show_progress_bar=False, convert_to_tensor=True, device=device)
            image_avg_embeddings.append(image_embs.mean(axis=0).cpu().numpy())
        else:
            # ==========================> 终极修复在这里 <==========================
            # 我们不再调用那个有bug的函数，直接使用已知的向量长度 512
            image_avg_embeddings.append([0] * 512)
            # ====================================================================

    bio_df = pd.DataFrame(bio_embeddings.cpu().numpy(), columns=[f'bio_vec_{i}' for i in range(bio_embeddings.shape[1])])
    title_df = pd.DataFrame(title_avg_embeddings, columns=[f'title_vec_{i}' for i in range(len(title_avg_embeddings[0]))])
    image_df = pd.DataFrame(image_avg_embeddings, columns=[f'image_vec_{i}' for i in range(len(image_avg_embeddings[0]))])
    
    df_features = df_features.drop(columns=['bio_text', 'note_titles', 'note_image_paths'])
    
    final_df = pd.concat([df_features.reset_index(drop=True), bio_df, title_df, image_df], axis=1)
    
    return final_df


def main():
    """主函数，执行整个流程"""
    # 1. 加载数据
    df_bloggers, df_notes = load_and_prepare_data()

    # 2. 计算基础特征
    df_features = feature_engineering(df_bloggers, df_notes)

    # 3. 使用AI模型进行向量化
    final_data = embed_features(df_features)

    # 4. 保存结果
    print(f"Step 5/5: 所有处理完成，正在保存结果到 {OUTPUT_CSV_FILE}...")
    final_data.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    print("="*30)
    print("🎉 恭喜！数据预处理成功！ 🎉")
    print(f"最终的训练文件 '{OUTPUT_CSV_FILE}' 已生成在您的项目文件夹中。")
    print("下一步，您可以将这个文件提供给我，来训练最终的分类模型。")
    print("="*30)


if __name__ == "__main__":
    main()

