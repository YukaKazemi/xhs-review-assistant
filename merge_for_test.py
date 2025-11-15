# merge_for_test.py (新版)
import pandas as pd

print("正在合并所有数据以进行最终验证...")

# 读取“符合”的文件，并添加原始标签
df_pos_blogger = pd.read_excel('已通过数据.xlsx', sheet_name='博主信息')
df_pos_blogger['原始审核状态'] = '符合' # 添加标准答案
df_pos_notes = pd.read_excel('已通过数据.xlsx', sheet_name='博主笔记')

# 读取“不符合”的文件，并添加原始标签
df_neg_blogger = pd.read_excel('未通过数据.xlsx', sheet_name='博主信息')
df_neg_blogger['原始审核状态'] = '不符合' # 添加标准答案
df_neg_notes = pd.read_excel('未通过数据.xlsx', sheet_name='博主笔记')

# 合并 '博主信息' sheet
df_blogger_all = pd.concat([df_pos_blogger, df_neg_blogger], ignore_index=True)

# 合并 '博主笔记' sheet
df_notes_all = pd.concat([df_pos_notes, df_neg_notes], ignore_index=True)

# 创建一个新的Excel写入器
with pd.ExcelWriter('全部测试数据.xlsx', engine='openpyxl') as writer:
    df_blogger_all.to_excel(writer, sheet_name='博主信息', index=False)
    df_notes_all.to_excel(writer, sheet_name='博主笔记', index=False)

print("成功！已将所有数据（包含原始审核状态）合并到 '全部测试数据.xlsx' 文件中。")
