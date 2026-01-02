# experiments/run_attack.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import jieba
import matplotlib
import time

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.models.svm_model import SVMModel
from src.attackers.wordvec_attacker import WordVectorAttacker


def run_progressive_attack_experiment():
    """运行渐进式对抗攻击实验"""
    print("=== 渐进式词向量对抗攻击实验 ===")
    start_time = time.time()

    # 1. 加载所有测试数据
    print("\n1. 加载所有测试数据...")
    test_df = pd.read_csv(CONFIG['TEST_CLEANED'])
    X_test = test_df['specific_dialogue_content'].astype(str).tolist()
    y_test = test_df['is_fraud'].astype(int).tolist()

    print(f"测试集总大小: {len(X_test)} 个样本")
    print(f"  欺诈样本: {sum(y_test)}")
    print(f"  非欺诈样本: {len(y_test) - sum(y_test)}")

    # 2. 加载模型
    print("\n2. 加载SVM受害模型...")
    model_path = CONFIG['ROOT_DIR'] + "/experiments/save_models/svm_model.pkl"
    svm = SVMModel.load(model_path)

    # 3. 原始模型性能
    print("\n3. 评估原始模型性能...")
    original_preds = svm.predict(X_test)
    original_acc = np.mean(np.array(original_preds) == np.array(y_test))

    print(f"原始模型准确率: {original_acc:.4f}")

    # 计算分类报告
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n原始模型分类报告:")
    print(classification_report(y_test, original_preds, target_names=['非欺诈', '欺诈']))

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, original_preds)
    print("混淆矩阵:")
    print(f"真阴性(TN): {cm[0, 0]}, 假阳性(FP): {cm[0, 1]}")
    print(f"假阴性(FN): {cm[1, 0]}, 真阳性(TP): {cm[1, 1]}")

    # 4. 渐进式攻击实验
    print("\n4. 运行渐进式词向量攻击...")

    # 设置词向量路径
    vector_path = CONFIG['ROOT_DIR'] + "/data/word_vectors/cc.zh.300.vec"

    if not os.path.exists(vector_path):
        print(f"警告: 词向量文件不存在: {vector_path}")
        print("将使用内置小型词向量")

    wordvec_attacker = WordVectorAttacker(svm, vector_path=vector_path)

    # 使用渐进式攻击
    adv_texts, attack_success_rate = wordvec_attacker.generate_adversarial_batch(
        X_test, y_test, max_attempts=3
    )

    # 分析结果
    results = wordvec_attacker.evaluate_attack(X_test, adv_texts, y_test)

    print(f"\n渐进式攻击结果:")
    print(f"  对抗样本准确率: {results['adversarial_accuracy']:.4f}")
    print(f"  准确率下降: {results['accuracy_drop']:.4f}")
    print(f"  攻击成功率: {results['attack_success_rate']:.4f}")
    print(f"  文本平均相似度: {results['avg_text_similarity']:.4f}")
    print(f"  成功攻击数: {results['successful_attacks']}/{results['total_fraud_samples']}")

    # 计算相对准确率下降
    relative_drop = (original_acc - results['adversarial_accuracy']) / original_acc * 100

    # 5. 保存结果
    print("\n5. 保存实验结果...")
    results_dir = CONFIG['ROOT_DIR'] + "/experiments/results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存详细结果
    results_df = pd.DataFrame({
        'original_text': X_test,
        'adversarial_text': adv_texts,
        'true_label': y_test,
        'original_pred': original_preds,
        'adversarial_pred': svm.predict(adv_texts)
    })

    results_path = os.path.join(results_dir, "progressive_adversarial_samples.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"对抗样本已保存到: {results_path}")

    # 保存统计结果
    stats = {
        'experiment_info': {
            'total_samples': len(X_test),
            'fraud_samples': sum(y_test),
            'non_fraud_samples': len(y_test) - sum(y_test),
            'method': 'progressive_wordvec_attack',
            'word_vector_size': 500000,
            'word_vector_dim': 300,
            'max_attempts': 3,
            'attack_strategy': 'progressive (10%, 20%, 30% replacement ratios)'
        },
        'original_performance': {
            'accuracy': float(original_acc),
            'confusion_matrix': {
                'TN': int(cm[0, 0]),
                'FP': int(cm[0, 1]),
                'FN': int(cm[1, 0]),
                'TP': int(cm[1, 1])
            }
        },
        'adversarial_attack': {
            'adversarial_accuracy': float(results['adversarial_accuracy']),
            'accuracy_drop': float(results['accuracy_drop']),
            'attack_success_rate': float(results['attack_success_rate']),
            'avg_text_similarity': float(results['avg_text_similarity']),
            'successful_attacks': int(results['successful_attacks']),
            'total_fraud_samples': int(results['total_fraud_samples']),
            'accuracy_relative_drop': float(relative_drop)
        }
    }

    stats_path = os.path.join(results_dir, "progressive_attack_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计结果已保存到: {stats_path}")

    # 6. 可视化
    print("\n6. 生成可视化图表...")
    plt.figure(figsize=(12, 5))

    # 子图1：准确率对比
    plt.subplot(1, 2, 1)
    methods = ['原始模型', '对抗样本']
    accuracies = [original_acc, results['adversarial_accuracy']]
    colors = ['blue', 'orange']

    bars = plt.bar(methods, accuracies, color=colors)
    plt.title('模型准确率对比', fontsize=14)
    plt.ylabel('准确率', fontsize=12)
    plt.ylim(0, 1.1)

    # 在柱子上添加数值
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=11)

    # 子图2：攻击效果
    plt.subplot(1, 2, 2)
    metrics = ['攻击成功率', '准确率下降', '相对准确率下降']
    values = [results['attack_success_rate'], results['accuracy_drop'], relative_drop / 100]
    colors = ['green', 'red', 'purple']

    bars = plt.bar(metrics, values, color=colors)
    plt.title('攻击效果评估', fontsize=14)
    plt.ylabel('数值', fontsize=12)
    plt.ylim(0, 1.1)

    for bar, val in zip(bars, values):
        if metrics[list(bars).index(bar)] == '相对准确率下降':
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val * 100:.1f}%', ha='center', va='bottom', fontsize=11)
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    chart_path = os.path.join(results_dir, "progressive_attack_results.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {chart_path}")

    # 7. 示例展示
    print("\n7. 对抗样本示例:")
    print("=" * 120)

    # 找出攻击成功的样本
    successful_samples = []
    for i in range(len(X_test)):
        if y_test[i] == 1:  # 欺诈样本
            orig_pred = original_preds[i]
            adv_pred = svm.predict([adv_texts[i]])[0]

            if orig_pred == 1 and adv_pred == 0:  # 攻击成功
                successful_samples.append(i)

    # 显示攻击成功的示例
    if successful_samples:
        print(f"找到 {len(successful_samples)} 个攻击成功的样本")

        for i, idx in enumerate(successful_samples[:3]):  # 最多显示3个
            print(f"\n示例 {i + 1} (攻击成功):")
            print(f"原始文本: {X_test[idx][:80]}...")
            print(f"对抗文本: {adv_texts[idx][:80]}...")

            # 计算相似度
            orig_words = set([w for w in jieba.lcut(X_test[idx]) if len(w.strip()) > 0])
            adv_words = set([w for w in jieba.lcut(adv_texts[idx]) if len(w.strip()) > 0])
            if orig_words and adv_words:
                similarity = len(orig_words & adv_words) / len(orig_words | adv_words)
                print(f"文本相似度: {similarity:.3f}")

            print(f"原始预测: {original_preds[idx]}, 攻击后预测: {svm.predict([adv_texts[idx]])[0]}")
            print("-" * 80)
    else:
        print("没有攻击成功的样本")

    total_time = time.time() - start_time
    print(f"\n实验总用时: {total_time:.1f}秒")

    return stats


def main():
    """主函数"""
    print("=== 渐进式对抗攻击实验 ===")

    # 检查词向量文件
    vector_path = CONFIG['ROOT_DIR'] + "/data/word_vectors/cc.zh.300.vec"

    if not os.path.exists(vector_path):
        print(f"警告: 词向量文件不存在: {vector_path}")
        print("\n将使用内置小型词向量继续实验...")

    # 运行实验
    try:
        results = run_progressive_attack_experiment()

        print("\n" + "=" * 80)
        print("实验完成！")
        print("=" * 80)

        # 打印关键结果
        print("\n关键实验结果:")
        print(f"1. 测试集大小: {results['experiment_info']['total_samples']}")
        print(f"2. 原始模型准确率: {results['original_performance']['accuracy']:.4f}")
        print(f"3. 对抗样本准确率: {results['adversarial_attack']['adversarial_accuracy']:.4f}")
        print(f"4. 准确率下降: {results['adversarial_attack']['accuracy_drop']:.4f}")
        print(f"5. 相对准确率下降: {results['adversarial_attack']['accuracy_relative_drop']:.1f}%")
        print(f"6. 攻击成功率: {results['adversarial_attack']['attack_success_rate']:.4f}")
        print(
            f"7. 成功攻击数: {results['adversarial_attack']['successful_attacks']}/{results['adversarial_attack']['total_fraud_samples']}")
        print(f"8. 文本平均相似度: {results['adversarial_attack']['avg_text_similarity']:.4f}")

        return results

    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
