# experiments/train_victim.py
import torch
from torch.optim import AdamW
from transformers import BertConfig
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import os
import logging
from config import CONFIG

# 导入自定义模块
from src import create_data_loaders
from src import BertClassifier

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(use_cleaned=True):
    """加载数据，优先使用清洗后的数据"""
    if use_cleaned:
        # 检查清洗后的数据是否存在
        if os.path.exists(CONFIG['TRAIN_CLEANED']):
            logger.info(f"使用清洗后的数据: {CONFIG['TRAIN_CLEANED']}")
            train_df = pd.read_csv(CONFIG['TRAIN_CLEANED'], encoding='utf-8-sig')
            test_df = pd.read_csv(CONFIG['TEST_CLEANED'], encoding='utf-8-sig')
        else:
            logger.warning(f"清洗后的数据不存在，使用原始数据")
            logger.info(f"请先运行数据清洗: python -c 'from src.utils.data_explorer import main; main()'")
            train_df = pd.read_csv(CONFIG['TRAIN_DATA'], encoding='utf-8-sig')
            test_df = pd.read_csv(CONFIG['TEST_DATA'], encoding='utf-8-sig')
    else:
        logger.info("使用原始数据")
        train_df = pd.read_csv(CONFIG['TRAIN_DATA'], encoding='utf-8-sig')
        test_df = pd.read_csv(CONFIG['TEST_DATA'], encoding='utf-8-sig')

    return train_df, test_df


def train_epoch(_model, data_loader, optimizer, device):
    """训练一个epoch"""
    _model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(data_loader):
        # 将数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        optimizer.zero_grad()
        loss, logits = _model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 每50个batch打印一次
        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    # 计算准确率
    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)

    return avg_loss, acc


def evaluate(_model, data_loader, device):
    """评估模型"""
    _model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            _, logits = _model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 获取预测结果
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算各种指标
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )

    return {
        'accuracy': acc,
        'precision_non_fraud': precision[0],
        'recall_non_fraud': recall[0],
        'f1_non_fraud': f1[0],
        'precision_fraud': precision[1],
        'recall_fraud': recall[1],
        'f1_fraud': f1[1],
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def print_evaluation_results(_results):
    """打印评估结果"""
    logger.info(f"总体准确率: {_results['accuracy']:.4f}")
    logger.info(f"\n非欺诈类:")
    logger.info(f"  精确率: {_results['precision_non_fraud']:.4f}")
    logger.info(f"  召回率: {_results['recall_non_fraud']:.4f}")
    logger.info(f"  F1分数: {_results['f1_non_fraud']:.4f}")
    logger.info(f"\n欺诈类:")
    logger.info(f"  精确率: {_results['precision_fraud']:.4f}")
    logger.info(f"  召回率: {_results['recall_fraud']:.4f}")
    logger.info(f"  F1分数: {_results['f1_fraud']:.4f}")


def main():
    """主训练函数"""
    logger.info("=== 开始训练欺诈检测模型 ===")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 从配置获取参数
    batch_size = int(CONFIG['BATCH_SIZE'])
    max_length = int(CONFIG['MAX_LENGTH'])
    learning_rate = float(CONFIG['LEARNING_RATE'])
    num_epochs = int(CONFIG['NUM_EPOCHS'])
    hidden_dropout_prob = int(CONFIG['HIDDEN_DROPOUT_PROB'])

    logger.info(f"配置参数:")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Max Length: {max_length}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Dropout: {hidden_dropout_prob}")

    # 加载数据
    logger.info("加载数据...")
    train_df, test_df = load_data(use_cleaned=True)

    # 数据预处理
    logger.info("数据预处理...")
    # 确保标签是整数
    train_df['is_fraud'] = train_df['is_fraud'].astype(int)
    test_df['is_fraud'] = test_df['is_fraud'].astype(int)

    # 移除空值
    train_df = train_df.dropna(subset=['specific_dialogue_content'])
    test_df = test_df.dropna(subset=['specific_dialogue_content'])

    # 数据统计
    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"测试集大小: {len(test_df)}")
    logger.info(f"训练集欺诈比例: {train_df['is_fraud'].mean():.2%}")
    logger.info(f"测试集欺诈比例: {test_df['is_fraud'].mean():.2%}")

    # 创建数据加载器
    train_loader, test_loader, _tokenizer = create_data_loaders(
        train_df, test_df, batch_size=batch_size
    )
    logger.info(f"训练集批次: {len(train_loader)}, 测试集批次: {len(test_loader)}")

    # 创建模型
    logger.info("初始化模型...")
    config = BertConfig.from_pretrained('bert-base-chinese', num_labels=2)
    config.hidden_dropout_prob = hidden_dropout_prob

    _model = BertClassifier.from_pretrained(
        'bert-base-chinese',
        config=config
    )
    _model.to(device)

    # 设置优化器
    optimizer = AdamW(_model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 训练循环
    logger.info(f"\n开始训练，共 {num_epochs} 个epoch")
    best_f1 = 0
    history = []

    for epoch in range(num_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            _model, train_loader, optimizer, device
        )

        # 评估
        eval_results = evaluate(_model, test_loader, device)

        # 计算训练时间
        epoch_time = time.time() - start_time

        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': eval_results['accuracy'],
            'f1_fraud': eval_results['f1_fraud'],
            'f1_non_fraud': eval_results['f1_non_fraud'],
            'time': epoch_time
        })

        # 打印结果
        logger.info(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        logger.info(f"测试准确率: {eval_results['accuracy']:.4f}")
        logger.info(f"非欺诈F1: {eval_results['f1_non_fraud']:.4f}, "
                    f"欺诈F1: {eval_results['f1_fraud']:.4f}")
        logger.info(f"时间: {epoch_time:.2f}秒")

        # 保存最佳模型
        avg_f1 = (eval_results['f1_fraud'] + eval_results['f1_non_fraud']) / 2
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            # 创建保存目录
            save_dir = "saved_models/victim_model"
            os.makedirs(save_dir, exist_ok=True)

            # 保存模型和tokenizer
            _model.save_pretrained(save_dir)
            _tokenizer.save_pretrained(save_dir)
            logger.info(f"保存最佳模型到: {save_dir}")

    logger.info(f"\n=== 训练完成 ===")
    logger.info(f"最佳平均F1分数: {best_f1:.4f}")

    # 最终评估
    logger.info(f"\n最终模型在测试集上的表现:")
    final_results = evaluate(_model, test_loader, device)
    print_evaluation_results(final_results)

    # 保存评估结果
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存预测结果
    results_df = pd.DataFrame({
        'true_label': final_results['labels'],
        'predicted_label': final_results['predictions'],
        'prob_fraud': [p[1] for p in final_results['probabilities']],
        'prob_non_fraud': [p[0] for p in final_results['probabilities']]
    })
    results_path = os.path.join(results_dir, 'victim_model_predictions.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n预测结果已保存到: {results_path}")

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_path = os.path.join(results_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    logger.info(f"训练历史已保存到: {history_path}")

    # 保存配置信息
    config_info = {
        'batch_size': batch_size,
        'max_length': max_length,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'hidden_dropout_prob': hidden_dropout_prob,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'final_accuracy': final_results['accuracy'],
        'final_f1_fraud': final_results['f1_fraud'],
        'final_f1_non_fraud': final_results['f1_non_fraud']
    }
    config_df = pd.DataFrame([config_info])
    config_path = os.path.join(results_dir, 'training_config.csv')
    config_df.to_csv(config_path, index=False)
    logger.info(f"训练配置已保存到: {config_path}")

    return _model, _tokenizer, final_results


if __name__ == "__main__":
    model, tokenizer, results = main()
