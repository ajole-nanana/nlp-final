# src/attackers/wordvec_attacker.py
import numpy as np
import jieba
import random
from typing import List, Tuple, Dict
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordVectorAttacker:
    """基于词向量的对抗样本生成器"""

    def __init__(self, victim_model, vector_path: str = None):
        """
        初始化攻击器

        Args:
            victim_model: 受害模型
            vector_path: 词向量文件路径
        """
        self.victim_model = victim_model

        # 加载词向量
        self.word_vectors = self._load_word_vectors(vector_path)

        # 设置jieba分词
        jieba.initialize()

        # 同义词缓存
        self.synonym_cache: Dict[str, List[str]] = {}

        logger.info(f"词向量攻击器初始化完成，词汇量: {len(self.word_vectors)}")

    def _load_word_vectors(self, vector_path: str) -> Dict[str, np.ndarray]:
        """
        加载词向量

        Args:
            vector_path: 词向量文件路径

        Returns:
            词向量字典
        """
        word_vectors = {}

        if not vector_path:
            # 如果没有提供路径，使用内置的小型词向量
            logger.warning("未提供词向量路径，使用内置小型词向量")
            return self._get_builtin_vectors()

        try:
            logger.info(f"正在加载词向量: {vector_path}")

            # 只加载前10000个词向量以加速
            max_words = 500000
            loaded_words = 0

            with open(vector_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 跳过第一行（元信息）
                first_line = f.readline().strip()

                # 读取词向量
                for line in f:
                    if loaded_words >= max_words:
                        break

                    parts = line.rstrip().split(' ')
                    if len(parts) < 10:  # 确保有足够的维度
                        continue

                    word = parts[0]

                    # 只加载中文词（简单过滤）
                    if self._is_chinese_word(word) and 2 <= len(word) <= 6:
                        try:
                            # 只取前50维以节省内存和计算时间
                            vector = np.array([float(x) for x in parts[1:301]])
                            word_vectors[word] = vector
                            loaded_words += 1
                        except (ValueError, IndexError):
                            continue

            logger.info(f"成功加载 {len(word_vectors)} 个词向量（限制为前{max_words}个）")
            return word_vectors

        except Exception as e:
            logger.error(f"加载词向量失败: {e}")
            logger.info("使用内置小型词向量")
            return self._get_builtin_vectors()

    def _is_chinese_word(self, word: str) -> bool:
        """判断是否为中文字符"""
        for char in word:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def _get_builtin_vectors(self) -> Dict[str, np.ndarray]:
        """获取内置的小型词向量（用于测试）"""
        vectors = {}

        # 与欺诈检测相关的关键词向量
        fraud_keywords = [
            '贷款', '借款', '融资', '放款', '借贷',
            '客服', '服务员', '服务人员', '客户服务',
            '银行', '金融机构', '钱庄', '储蓄所',
            '诈骗', '欺诈', '骗局', '欺骗',
            '退款', '返款', '退钱', '退还', '返还',
            '账户', '账号', '户头', '银行账户',
            '密码', '口令', '密钥', '暗码',
            '链接', '网址', 'URL', '网站地址',
            '验证码', '校验码', '识别码', '验证代码',
            '转账', '汇款', '划款', '转款',
            '投资', '出资', '投入', '融资',
            '收益', '利润', '回报', '收入',
            '风险', '危险', '风险系数', '不确定性',
            '安全', '保险', '可靠', '稳妥',
            '优惠', '折扣', '特价', '便宜',
            '申请', '申报', '请求', '提请',
            '确认', '确定', '认可', '证实',
            '操作', '处理', '执行', '进行',
            '系统', '体系', '系统软件', '平台',
            '信息', '资料', '消息', '情报',
            '电话', '手机', '座机', '电话号码',
            '需要', '要求', '需求', '必需',
            '问题', '疑问', '难题', '麻烦',
            '帮助', '协助', '支援', '帮忙',
        ]

        dim = 50  # 小型向量的维度
        np.random.seed(42)  # 固定随机种子

        for word in fraud_keywords:
            vectors[word] = np.random.randn(dim)

        return vectors

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0.0

        # 归一化
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _get_similar_words_fast(self, word: str, top_n: int = 5) -> List[str]:
        """
        快速获取相似词（使用缓存和简单规则）

        Args:
            word: 目标词
            top_n: 返回的相似词数量

        Returns:
            相似词列表
        """
        # 检查缓存
        if word in self.synonym_cache:
            return self.synonym_cache[word][:top_n]

        # 如果词不在词向量中，返回空列表
        if word not in self.word_vectors:
            self.synonym_cache[word] = []
            return []

        # 简单的同义词规则（无需遍历整个词向量）
        synonym_rules = {
            '贷款': ['借款', '融资', '放款'],
            '客服': ['服务人员', '服务员', '专员'],
            '银行': ['金融机构', '银行机构'],
            '诈骗': ['欺诈', '骗局', '欺骗'],
            '退款': ['返款', '退钱', '退还'],
            '账户': ['账号', '户头'],
            '密码': ['口令', '密钥'],
            '链接': ['网址', '连接'],
            '验证码': ['校验码', '识别码'],
            '转账': ['汇款', '划款'],
            '投资': ['出资', '投入'],
            '收益': ['利润', '回报'],
            '风险': ['危险', '不确定性'],
            '安全': ['保险', '可靠'],
            '优惠': ['折扣', '特价'],
            '申请': ['申报', '请求'],
            '确认': ['确定', '认可'],
            '操作': ['处理', '执行'],
            '系统': ['体系', '平台'],
            '信息': ['资料', '消息'],
            '电话': ['手机', '座机'],
            '需要': ['要求', '需求'],
            '问题': ['疑问', '难题'],
            '帮助': ['协助', '支援'],
        }

        # 从规则中获取同义词
        if word in synonym_rules:
            synonyms = synonym_rules[word]
        else:
            # 对于不在规则中的词，返回一些常见的替代词
            common_replacements = ['相关', '相应', '有关', '对应', '这个', '那个']
            synonyms = random.sample(common_replacements, min(3, len(common_replacements)))

        # 缓存结果
        self.synonym_cache[word] = synonyms

        return synonyms[:top_n]

    def extract_keywords_simple(self, text: str) -> List[str]:
        """
        简单关键词提取

        Args:
            text: 输入文本

        Returns:
            关键词列表
        """
        words = list(jieba.cut(text))

        # 过滤词
        filtered_words = []
        for word in words:
            word = word.strip()
            if len(word) >= 2:  # 只保留长度>=2的词
                filtered_words.append(word)

        # 返回前10个词作为关键词
        return filtered_words[:10]

    def replace_with_synonyms_simple(self, text: str, replace_ratio: float = 0.3) -> str:
        """
        使用简单规则的同义词替换

        Args:
            text: 原始文本
            replace_ratio: 替换比例（0-1）

        Returns:
            对抗样本文本
        """
        words = list(jieba.cut(text))

        if len(words) < 3:  # 文本太短，不进行替换
            return text

        # 计算要替换的词数
        n_replace = max(1, int(len(words) * replace_ratio))

        # 选择要替换的词（选择长度>=2的词）
        candidate_indices = []
        for i, word in enumerate(words):
            if len(word.strip()) >= 2:
                candidate_indices.append(i)

        # 随机选择要替换的词
        if len(candidate_indices) > 0:
            n_to_replace = min(n_replace, len(candidate_indices))
            selected_indices = random.sample(candidate_indices, n_to_replace)
        else:
            selected_indices = []

        # 进行替换
        new_words = words.copy()
        for idx in selected_indices:
            if idx >= len(new_words):
                continue

            original_word = words[idx].strip()

            # 获取相似词
            similar_words = self._get_similar_words_fast(original_word, top_n=3)

            if similar_words:
                # 随机选择一个相似词
                new_word = random.choice(similar_words)
                new_words[idx] = new_word

        return ''.join(new_words)

    def generate_adversarial_fast(self, texts: List[str], labels: List[int],
                                  target_label: int = 0) -> Tuple[List[str], float]:
        """
        快速生成对抗样本

        Args:
            texts: 原始文本列表
            labels: 标签列表
            target_label: 目标标签

        Returns:
            对抗样本列表，攻击成功率
        """
        logger.info(f"开始快速生成对抗样本，目标标签: {target_label}")

        adversarial_texts = []
        success_count = 0
        total_attacks = sum(1 for label in labels if label == 1)

        if total_attacks == 0:
            logger.warning("没有欺诈样本需要攻击")
            return texts, 0.0

        start_time = time.time()

        for i, (text, label) in enumerate(zip(texts, labels)):
            if i % 20 == 0:
                elapsed = time.time() - start_time
                logger.info(f"处理进度: {i}/{len(texts)} (已用时: {elapsed:.1f}秒)")

            if label == 1:  # 只攻击欺诈样本
                original_pred = self.victim_model.predict([text])[0]

                # 如果原始预测已经是目标标签，跳过
                if original_pred == target_label:
                    adversarial_texts.append(text)
                    continue

                # 生成对抗样本（使用简单方法）
                adv_text = self.replace_with_synonyms_simple(text, replace_ratio=0.3)

                # 获取预测结果
                adv_pred = self.victim_model.predict([adv_text])[0]

                if adv_pred == target_label:
                    adversarial_texts.append(adv_text)
                    success_count += 1
                else:
                    # 攻击失败，保留原始文本
                    adversarial_texts.append(text)
            else:
                # 非欺诈样本保持不变
                adversarial_texts.append(text)

        attack_success_rate = success_count / total_attacks if total_attacks > 0 else 0.0
        total_time = time.time() - start_time

        logger.info(f"攻击完成，成功率: {attack_success_rate:.2%} ({success_count}/{total_attacks})")
        logger.info(f"总用时: {total_time:.1f}秒，平均每个样本: {total_time / len(texts):.3f}秒")

        return adversarial_texts, attack_success_rate

    def evaluate_attack(self, original_texts: List[str],
                        adversarial_texts: List[str],
                        labels: List[int]) -> Dict:
        """
        评估攻击效果

        Args:
            original_texts: 原始文本
            adversarial_texts: 对抗文本
            labels: 真实标签

        Returns:
            评估结果字典
        """
        # 获取预测结果
        orig_preds = self.victim_model.predict(original_texts)
        adv_preds = self.victim_model.predict(adversarial_texts)

        # 计算准确率
        orig_acc = np.mean(np.array(orig_preds) == np.array(labels))
        adv_acc = np.mean(np.array(adv_preds) == np.array(labels))

        # 计算攻击成功率（只考虑欺诈样本）
        fraud_indices = [i for i, label in enumerate(labels) if label == 1]
        if fraud_indices:
            orig_fraud_preds = [orig_preds[i] for i in fraud_indices]
            adv_fraud_preds = [adv_preds[i] for i in fraud_indices]

            # 攻击成功：欺诈样本被预测为非欺诈（目标标签0）
            attack_success = sum(1 for pred in adv_fraud_preds if pred == 0)
            total_fraud = len(fraud_indices)
            attack_success_rate = attack_success / total_fraud if total_fraud > 0 else 0
        else:
            attack_success_rate = 0

        # 计算文本相似度（简单的词重叠）
        similarities = []
        for i, (orig, adv) in enumerate(zip(original_texts, adversarial_texts)):
            # 只计算欺诈样本的相似度
            if labels[i] == 1:
                orig_words = set([w for w in jieba.lcut(orig) if len(w.strip()) > 0])
                adv_words = set([w for w in jieba.lcut(adv) if len(w.strip()) > 0])

                if orig_words and adv_words:
                    intersection = len(orig_words & adv_words)
                    union = len(orig_words | adv_words)
                    if union > 0:
                        similarity = intersection / union
                        similarities.append(similarity)

        avg_similarity = np.mean(similarities) if similarities else 0

        return {
            'original_accuracy': float(orig_acc),
            'adversarial_accuracy': float(adv_acc),
            'accuracy_drop': float(orig_acc - adv_acc),
            'attack_success_rate': float(attack_success_rate),
            'avg_text_similarity': float(avg_similarity),
            'total_fraud_samples': len(fraud_indices),
            'successful_attacks': int(attack_success) if 'attack_success' in locals() else 0
        }


def test_wordvec_attacker():
    """测试词向量攻击器"""
    import pandas as pd
    from config import CONFIG
    from src.models.svm_model import SVMModel

    # 加载模型
    model_path = CONFIG['ROOT_DIR'] + "/experiments/save_models/svm_model.pkl"
    svm = SVMModel.load(model_path)

    # 加载测试数据
    test_df = pd.read_csv(CONFIG['TEST_CLEANED'])

    # 使用少量数据进行测试
    test_fraud = test_df[test_df['is_fraud'] == 1].sample(5, random_state=42)
    test_non_fraud = test_df[test_df['is_fraud'] == 0].sample(5, random_state=42)

    test_subset = pd.concat([test_fraud, test_non_fraud])
    X_test = test_subset['specific_dialogue_content'].astype(str).tolist()
    y_test = test_subset['is_fraud'].astype(int).tolist()

    print(f"测试数据: {len(X_test)} 个样本 (欺诈: {sum(y_test)})")

    # 创建攻击器
    attacker = WordVectorAttacker(svm, vector_path=CONFIG['ROOT_DIR'] + "/data/word_vectors/cc.zh.300.vec")

    # 生成对抗样本
    adv_texts, success_rate = attacker.generate_adversarial_fast(
        X_test, y_test, target_label=0
    )

    # 分析结果
    results = attacker.evaluate_attack(X_test, adv_texts, y_test)

    print("\n=== 攻击结果 ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # 显示示例
    print("\n=== 对抗样本示例 ===")
    for i in range(min(3, len(X_test))):
        if y_test[i] == 1:
            print(f"\n示例 {i + 1}:")
            print(f"原始文本: {X_test[i][:80]}...")
            print(f"对抗文本: {adv_texts[i][:80]}...")

            orig_pred = svm.predict([X_test[i]])[0]
            adv_pred = svm.predict([adv_texts[i]])[0]
            print(f"原始预测: {orig_pred}, 对抗预测: {adv_pred}")


if __name__ == "__main__":
    test_wordvec_attacker()