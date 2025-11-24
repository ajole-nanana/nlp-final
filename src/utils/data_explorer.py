"""
数据探索和清洗工具
1. 简单的数据质量分析
2. 去掉不利于训练的部分字段为空的数据
"""
import pandas as pd
import os
from config import CONFIG


class DataExplorer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None

    def load_data(self):
        """加载训练和测试数据"""
        print("正在加载数据...")

        # 检查文件是否存在
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"训练数据文件不存在: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"测试数据文件不存在: {self.test_path}")

        # 读取CSV，处理可能的编码问题
        encodings = ['utf-8', 'gbk', 'utf-8-sig']

        for encoding in encodings:
            try:
                self.train_df = pd.read_csv(self.train_path, encoding=encoding)
                self.test_df = pd.read_csv(self.test_path, encoding=encoding)
                print(f"成功使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("无法读取文件，尝试了所有编码格式")

        print(f"训练集大小: {len(self.train_df)}")
        print(f"测试集大小: {len(self.test_df)}")

        # 显示列信息
        print(f"数据列: {list(self.train_df.columns)}")

    def preprocess_data(self):
        """预处理数据"""
        print("\n=== 数据预处理 ===")

        for df_name, df in [("训练集", self.train_df), ("测试集", self.test_df)]:
            print(f"\n处理 {df_name}:")

            # 处理 is_fraud 列 - 确保是数值类型
            df['is_fraud'] = df['is_fraud'].astype(str).str.upper().str.strip()
            df['is_fraud'] = df['is_fraud'].map({'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0, '1': 1, '0': 0})

            # 处理空值，将无法转换的设为0
            df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce').fillna(0).astype(int)

            print(f"欺诈标签分布: {dict(df['is_fraud'].value_counts())}")

    def explore_data_quality(self):
        """探索数据质量"""
        print("\n=== 数据质量分析 ===")

        for df_name, df in [("训练集", self.train_df), ("测试集", self.test_df)]:
            print(f"\n{df_name}:")
            print(f"总行数: {len(df)}")

            # 检查各列的空值情况
            for col in df.columns:
                null_count = df[col].isnull().sum()
                null_percent = (null_count / len(df)) * 100
                print(f"  {col}: {null_count} 空值 ({null_percent:.2f}%)")

    def analyze_fraud_distribution(self):
        """分析欺诈标签分布"""
        print("\n=== 欺诈标签分布 ===")

        for df_name, df in [("训练集", self.train_df), ("测试集", self.test_df)]:
            print(f"\n{df_name}:")
            fraud_counts = df['is_fraud'].value_counts()
            print(f"欺诈分布: {dict(fraud_counts)}")

            if len(fraud_counts) == 2:
                balance_ratio = min(fraud_counts) / max(fraud_counts)
                print(f"数据平衡比例: {balance_ratio:.3f}")

    def clean_data_simple(self):
        """简单有效的数据清洗"""
        print("\n=== 执行数据清洗 ===")

        original_train_size = len(self.train_df)
        original_test_size = len(self.test_df)

        cleaning_steps = []

        # 1. 移除 specific_dialogue_content 为空的行
        train_before = len(self.train_df)
        test_before = len(self.test_df)

        self.train_df = self.train_df[self.train_df['specific_dialogue_content'].notna()]
        self.test_df = self.test_df[self.test_df['specific_dialogue_content'].notna()]

        cleaning_steps.append({
            'step': '移除 specific_dialogue_content 为空的行',
            'train_removed': train_before - len(self.train_df),
            'test_removed': test_before - len(self.test_df)
        })

        # 2. 移除 is_fraud 为空的行
        train_before = len(self.train_df)
        test_before = len(self.test_df)

        self.train_df = self.train_df[self.train_df['is_fraud'].notna()]
        self.test_df = self.test_df[self.test_df['is_fraud'].notna()]

        cleaning_steps.append({
            'step': '移除 is_fraud 为空的行',
            'train_removed': train_before - len(self.train_df),
            'test_removed': test_before - len(self.test_df)
        })

        # 3. 对于其他列，如果空值比例小于10%，移除空值行；否则用'unknown'填充
        other_columns = ['call_type', 'interaction_strategy']

        for col in other_columns:
            if col in self.train_df.columns:
                # 训练集处理
                train_null_pct = self.train_df[col].isnull().sum() / len(self.train_df)
                if 0.1 > train_null_pct > 0:
                    train_before = len(self.train_df)
                    self.train_df = self.train_df[self.train_df[col].notna()]
                    cleaning_steps.append({
                        'step': f'移除训练集 {col} 为空的行',
                        'train_removed': train_before - len(self.train_df),
                        'test_removed': 0
                    })
                else:
                    self.train_df[col] = self.train_df[col].fillna('unknown')

                # 测试集处理
                test_null_pct = self.test_df[col].isnull().sum() / len(self.test_df)
                if 0.1 > test_null_pct > 0:
                    test_before = len(self.test_df)
                    self.test_df = self.test_df[self.test_df[col].notna()]
                    cleaning_steps.append({
                        'step': f'移除测试集 {col} 为空的行',
                        'train_removed': 0,
                        'test_removed': test_before - len(self.test_df)
                    })
                else:
                    self.test_df[col] = self.test_df[col].fillna('unknown')

        # 4. 处理 fraud_type
        if 'fraud_type' in self.train_df.columns:
            # 对于非欺诈样本，fraud_type设为'non_fraud'
            self.train_df.loc[self.train_df['is_fraud'] == 0, 'fraud_type'] = 'non_fraud'
            self.test_df.loc[self.test_df['is_fraud'] == 0, 'fraud_type'] = 'non_fraud'

            # 对于欺诈样本但fraud_type为空的，设为'unknown_fraud'
            self.train_df['fraud_type'] = self.train_df['fraud_type'].fillna('unknown_fraud')
            self.test_df['fraud_type'] = self.test_df['fraud_type'].fillna('unknown_fraud')

        # 打印清洗报告
        print("\n=== 数据清洗报告 ===")
        for step in cleaning_steps:
            print(f"{step['step']}:")
            if step['train_removed'] > 0:
                print(f"  训练集移除: {step['train_removed']} 行")
            if step['test_removed'] > 0:
                print(f"  测试集移除: {step['test_removed']} 行")

        print(f"\n最终数据大小:")
        print(f"训练集: {len(self.train_df)} 行 (保留 {len(self.train_df) / original_train_size * 100:.1f}%)")
        print(f"测试集: {len(self.test_df)} 行 (保留 {len(self.test_df) / original_test_size * 100:.1f}%)")

        return self.train_df, self.test_df

    def save_cleaned_data(self, output_dir=CONFIG['PROCESSED']+"/cleaned"):
        """保存清洗后的数据 - 确保正确编码"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存为UTF-8编码，确保中文字符正确
        try:
            self.train_df.to_csv(f"{output_dir}/train_cleaned.csv", index=False, encoding='utf-8-sig')
            self.test_df.to_csv(f"{output_dir}/test_cleaned.csv", index=False, encoding='utf-8-sig')
            print(f"清洗后的数据已保存到: {output_dir} (UTF-8-BOM编码)")
        except Exception as e:
            print(f"保存UTF-8-BOM失败: {e}")
            # 回退到普通UTF-8
            self.train_df.to_csv(f"{output_dir}/train_cleaned.csv", index=False, encoding='utf-8')
            self.test_df.to_csv(f"{output_dir}/test_cleaned.csv", index=False, encoding='utf-8')
            print(f"清洗后的数据已保存到: {output_dir} (UTF-8编码)")


def main():
    """主函数 - 数据预处理流程"""
    print("=== 欺诈对话检测数据预处理 ===")

    # 数据路径
    train_path = CONFIG['TRAIN_DATA']
    test_path = CONFIG['TEST_DATA']

    try:
        # 初始化探索器
        explorer = DataExplorer(train_path, test_path)

        # 执行完整流程
        explorer.load_data()
        explorer.preprocess_data()
        explorer.explore_data_quality()
        explorer.analyze_fraud_distribution()

        # 执行清洗
        _clean_train_df, _clean_test_df = explorer.clean_data_simple()

        # 保存清洗后的数据
        explorer.save_cleaned_data()

        print("\n=== 预处理完成 ===")
        print(f"训练集最终大小: {len(_clean_train_df)}")
        print(f"测试集最终大小: {len(_clean_test_df)}")

        return _clean_train_df, _clean_test_df

    except Exception as e:
        print(f"数据处理过程中出现错误: {e}")
        return None, None


if __name__ == "__main__":
    clean_train_df, clean_test_df = main()
