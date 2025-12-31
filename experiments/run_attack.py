# experiments/run_attack.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import jieba
import matplotlib

# 设置中文字体，避免中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from src.models.svm_model import SVMModel
from src.attackers.wordvec_attacker import WordVectorAttacker


def run_full_dataset_experiment():
    """运行全数据集实验"""
    print("=== 全数据集词向量对抗攻击实验 ===")

    # 1. 加载所有测试数据
    print("\n1. 加载所有测试数据...")
    test_df = pd.read_csv(CONFIG['TEST_CLEANED'])

    # 使用所有测试数据
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

    # 4. 词向量攻击实验
    print("\n4. 运行词向量攻击（全数据集）...")

    # 设置词向量路径
    vector_path = CONFIG['ROOT_DIR'] + "/data/word_vectors/cc.zh.300.vec"

    if not os.path.exists(vector_path):
        print(f"警告: 词向量文件不存在: {vector_path}")
        print("将使用内置小型词向量")

    wordvec_attacker = WordVectorAttacker(svm, vector_path=vector_path)

    # 生成对抗样本（使用快速方法）
    adv_texts_wordvec, attack_success_rate = wordvec_attacker.generate_adversarial_fast(
        X_test, y_test, target_label=0
    )

    # 分析结果
    results_wordvec = wordvec_attacker.evaluate_attack(X_test, adv_texts_wordvec, y_test)

    print(f"\n词向量攻击结果:")
    print(f"  对抗样本准确率: {results_wordvec['adversarial_accuracy']:.4f}")
    print(f"  准确率下降: {results_wordvec['accuracy_drop']:.4f}")
    print(f"  攻击成功率: {results_wordvec['attack_success_rate']:.4f}")
    print(f"  文本平均相似度: {results_wordvec['avg_text_similarity']:.4f}")
    print(f"  成功攻击数: {results_wordvec['successful_attacks']}/{results_wordvec['total_fraud_samples']}")

    # 5. 保存结果
    print("\n5. 保存实验结果...")
    results_dir = CONFIG['ROOT_DIR'] + "/experiments/results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存详细结果
    results_df = pd.DataFrame({
        'original_text': X_test,
        'adversarial_text': adv_texts_wordvec,
        'true_label': y_test,
        'original_pred': original_preds,
        'adversarial_pred': svm.predict(adv_texts_wordvec)
    })

    results_path = os.path.join(CONFIG['ROOT_DIR'] + "/data/adversarial", "full_dataset_adversarial_samples.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"对抗样本已保存到: {results_path}")

    # 保存统计结果
    stats = {
        'experiment_info': {
            'total_samples': len(X_test),
            'fraud_samples': sum(y_test),
            'non_fraud_samples': len(y_test) - sum(y_test),
            'method': 'fast_wordvec_attack',
            'dataset': 'full_test_set'
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
            'adversarial_accuracy': float(results_wordvec['adversarial_accuracy']),
            'accuracy_drop': float(results_wordvec['accuracy_drop']),
            'attack_success_rate': float(results_wordvec['attack_success_rate']),
            'avg_text_similarity': float(results_wordvec['avg_text_similarity']),
            'successful_attacks': int(results_wordvec['successful_attacks']),
            'total_fraud_samples': int(results_wordvec['total_fraud_samples']),
            'accuracy_relative_drop': float(
                (original_acc - results_wordvec['adversarial_accuracy']) / original_acc * 100)
        }
    }

    stats_path = os.path.join(results_dir, "full_dataset_attack_statistics.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计结果已保存到: {stats_path}")

    # 6. 可视化
    print("\n6. 生成可视化图表...")
    plt.figure(figsize=(15, 5))

    # 子图1：准确率对比
    plt.subplot(1, 3, 1)
    methods = ['原始模型', '对抗样本']
    accuracies = [original_acc, results_wordvec['adversarial_accuracy']]
    colors = ['blue', 'orange']

    bars = plt.bar(methods, accuracies, color=colors)
    plt.title('模型准确率对比（全数据集）', fontsize=14)
    plt.ylabel('准确率', fontsize=12)
    plt.ylim(0, 1.1)

    # 在柱子上添加数值
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=11)

    # 子图2：攻击效果
    plt.subplot(1, 3, 2)
    metrics = ['攻击成功率', '准确率下降']
    values = [results_wordvec['attack_success_rate'], results_wordvec['accuracy_drop']]
    colors = ['green', 'red']

    bars = plt.bar(metrics, values, color=colors)
    plt.title('攻击效果评估', fontsize=14)
    plt.ylabel('数值', fontsize=12)
    plt.ylim(0, 1.1)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11)

    # 子图3：性能对比
    plt.subplot(1, 3, 3)
    performance_metrics = ['相对准确率下降']
    performance_values = [stats['adversarial_attack']['accuracy_relative_drop']]

    bars = plt.bar(performance_metrics, performance_values, color='purple')
    plt.title('性能下降百分比', fontsize=14)
    plt.ylabel('百分比 (%)', fontsize=12)

    for bar, val in zip(bars, performance_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    chart_path = os.path.join(results_dir, "full_dataset_attack_results.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {chart_path}")

    # 7. 详细分析报告
    print("\n7. 生成详细分析报告...")

    # 找出攻击成功的样本
    successful_samples = []
    for i in range(len(X_test)):
        if y_test[i] == 1:  # 欺诈样本
            orig_pred = original_preds[i]
            adv_pred = svm.predict([adv_texts_wordvec[i]])[0]

            if orig_pred == 1 and adv_pred == 0:  # 攻击成功
                successful_samples.append(i)

    # 找出攻击失败的样本（欺诈样本但攻击失败）
    failed_samples = []
    for i in range(len(X_test)):
        if y_test[i] == 1:  # 欺诈样本
            orig_pred = original_preds[i]
            adv_pred = svm.predict([adv_texts_wordvec[i]])[0]

            if orig_pred == 1 and adv_pred == 1:  # 攻击失败
                failed_samples.append(i)

    # 生成分析报告
    analysis_report = {
        'summary': {
            'total_samples': len(X_test),
            'fraud_samples': sum(y_test),
            'non_fraud_samples': len(y_test) - sum(y_test),
            'successful_attacks': len(successful_samples),
            'failed_attacks': len(failed_samples),
            'attack_success_rate': len(successful_samples) / sum(y_test) if sum(y_test) > 0 else 0
        },
        'attack_success_examples': [],
        'attack_failed_examples': []
    }

    # 添加攻击成功的示例
    for idx in successful_samples[:5]:  # 最多5个示例
        analysis_report['attack_success_examples'].append({
            'index': idx,
            'original_text_preview': X_test[idx][:100] + "..." if len(X_test[idx]) > 100 else X_test[idx],
            'adversarial_text_preview': adv_texts_wordvec[idx][:100] + "..." if len(adv_texts_wordvec[idx]) > 100 else
            adv_texts_wordvec[idx],
            'original_prediction': int(original_preds[idx]),
            'adversarial_prediction': int(svm.predict([adv_texts_wordvec[idx]])[0]),
            'text_similarity': _calculate_similarity(X_test[idx], adv_texts_wordvec[idx])
        })

    # 添加攻击失败的示例
    for idx in failed_samples[:5]:  # 最多5个示例
        analysis_report['attack_failed_examples'].append({
            'index': idx,
            'original_text_preview': X_test[idx][:100] + "..." if len(X_test[idx]) > 100 else X_test[idx],
            'adversarial_text_preview': adv_texts_wordvec[idx][:100] + "..." if len(adv_texts_wordvec[idx]) > 100 else
            adv_texts_wordvec[idx],
            'original_prediction': int(original_preds[idx]),
            'adversarial_prediction': int(svm.predict([adv_texts_wordvec[idx]])[0]),
            'text_similarity': _calculate_similarity(X_test[idx], adv_texts_wordvec[idx])
        })

    # 保存分析报告
    analysis_path = os.path.join(results_dir, "full_dataset_analysis_report.json")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, ensure_ascii=False, indent=2)
    print(f"分析报告已保存到: {analysis_path}")

    # 8. 示例展示
    print("\n8. 对抗样本示例分析:")
    print("=" * 120)

    if successful_samples:
        print(f"\n攻击成功示例（共{len(successful_samples)}个）:")

        for i, idx in enumerate(successful_samples[:3]):  # 最多显示3个
            print(f"\n示例 {i + 1} (索引: {idx}):")
            print(f"原始文本: {X_test[idx][:80]}...")
            print(f"对抗文本: {adv_texts_wordvec[idx][:80]}...")

            # 计算文本变化
            orig_words = set([w for w in jieba.lcut(X_test[idx]) if len(w.strip()) > 0])
            adv_words = set([w for w in jieba.lcut(adv_texts_wordvec[idx]) if len(w.strip()) > 0])
            changed_words = adv_words - orig_words
            common_words = adv_words & orig_words

            print(f"原始预测: {original_preds[idx]}, 对抗预测: {svm.predict([adv_texts_wordvec[idx]])[0]}")
            print(f"共同词数: {len(common_words)}, 变化词数: {len(changed_words)}")
            if changed_words:
                print(f"主要替换词: {list(changed_words)[:5]}")
            print("-" * 80)
    else:
        print("\n没有攻击成功的样本")

    print("\n" + "=" * 120)

    return stats


def _calculate_similarity(text1, text2):
    """计算两个文本的相似度"""
    words1 = set([w for w in jieba.lcut(text1) if len(w.strip()) > 0])
    words2 = set([w for w in jieba.lcut(text2) if len(w.strip()) > 0])

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def run_experiment_with_progress():
    """运行实验并显示进度"""
    print("=== 全数据集对抗攻击实验 ===")

    # 检查词向量文件
    vector_path = CONFIG['ROOT_DIR'] + "/data/word_vectors/cc.zh.300.vec"

    if not os.path.exists(vector_path):
        print(f"警告: 词向量文件不存在: {vector_path}")
        print("\n将使用内置小型词向量继续实验...")

    # 运行实验
    try:
        results = run_full_dataset_experiment()

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
    run_experiment_with_progress()