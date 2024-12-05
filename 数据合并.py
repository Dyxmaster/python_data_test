import pandas as pd

# 读取两个 CSV 文件
df1 = pd.read_csv('./data/real.csv',header=None)
df2 = pd.read_csv('./data/imag.csv',header=None)

# 检查两个文件是否具有相同的行数
if len(df1) != len(df2):
    raise ValueError("两个文件的行数不一致，无法进行拼接")

# 按列拼接两个 DataFrame
df_combined = pd.concat([df1, df2], axis=1)

print(df_combined.shape)
# 保存拼接后的文件
df_combined.to_csv('./data/combined_file.csv', index=False,header=False )

print("拼接完成，已保存")
