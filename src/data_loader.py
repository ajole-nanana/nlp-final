# src/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import CONFIG
from utils import DataExplorer


class FraudDataset(Dataset):
    """欺诈检测数据集类"""

    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 提取文本和标签
        text = str(self.data.iloc[idx]['specific_dialogue_content'])
        label = int(self.data.iloc[idx]['is_fraud'])

        # Tokenize文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(_train_df, _test_df, batch_size=16):
    """创建训练和测试数据加载器"""
    # 使用中文BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 创建数据集
    train_dataset = FraudDataset(_train_df, tokenizer)
    test_dataset = FraudDataset(_test_df, tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, tokenizer


def data_cleaned():
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
    train_df, test_df = data_cleaned()
