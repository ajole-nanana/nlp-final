# src/attackers/wordvec_attacker.py
import numpy as np
import jieba
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordVectorAttacker:
    """基于词向量的对抗样本生成器"""

    def __init__(self, victim_model, vector_path: str = None):
        self.victim_model = victim_model
        self.word_vectors = self._load_word_vectors(vector_path)
        jieba.initialize()
        self.synonym_cache = {}

        logger.info(f"词向量攻击器初始化完成，词汇量: {len(self.word_vectors)}")

    def _load_word_vectors(self, vector_path: str) -> Dict[str, np.ndarray]:
        """加载词向量"""
        word_vectors = {}

        if not vector_path:
            logger.warning("未提供词向量路径，使用内置小型词向量")
            return self._get_builtin_vectors()

        try:
            logger.info(f"正在加载词向量: {vector_path}")

            max_words = 5000  # 限制为50万词
            loaded_words = 0

            with open(vector_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 跳过第一行（元信息）
                first_line = f.readline()

                for line in f:
                    if loaded_words >= max_words:
                        break

                    parts = line.rstrip().split(' ')
                    if len(parts) < 10:
                        continue

                    word = parts[0]

                    # 只加载中文词
                    if self._is_chinese_word(word) and 2 <= len(word) <= 6:
                        try:
                            vector = np.array([float(x) for x in parts[1:151]])  # 前300维
                            word_vectors[word] = vector
                            loaded_words += 1
                        except (ValueError, IndexError):
                            continue

            logger.info(f"成功加载 {len(word_vectors)} 个词向量")
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
        """获取内置的小型词向量"""
        vectors = {}
        fraud_keywords = [
            '贷款', '借款', '融资', '放款', '借贷', '客服', '服务员', '银行',
            '诈骗', '欺诈', '退款', '账户', '密码', '链接', '验证码', '转账'
        ]

        dim = 100
        np.random.seed(42)

        for word in fraud_keywords:
            vectors[word] = np.random.randn(dim)

        return vectors

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0.0

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _get_similar_words(self, word: str, top_n: int = 3) -> List[str]:
        """获取相似词"""
        if word not in self.word_vectors:
            return []

        target_vector = self.word_vectors[word]
        similarities = []

        for candidate_word, candidate_vector in self.word_vectors.items():
            if candidate_word == word:
                continue

            similarity = self._cosine_similarity(target_vector, candidate_vector)
            if similarity > 0.6:  # 相似度阈值
                if abs(len(candidate_word) - len(word)) <= 2:  # 长度约束
                    similarities.append((candidate_word, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [synonym for synonym, _ in similarities[:top_n]]

    def _calculate_word_importance(self, words: List[str]) -> Dict[int, float]:
        """计算每个词的重要性分数"""
        importance_scores = {}

        for i, word in enumerate(words):
            word = word.strip()
            if len(word) < 2:
                continue

            score = 0

            # 词频分数
            score += words.count(word) * 1.0

            # 词长分数
            if len(word) >= 3:
                score += 2.0
            elif len(word) == 2:
                score += 1.0

            # 词向量存在性
            if word in self.word_vectors:
                score += 3.0

            # 欺诈相关性
            fraud_keywords = ['贷款', '客服', '银行', '诈骗', '退款', '账户',
                              '密码', '链接', '验证码', '转账', '投资', '安全']
            if word in fraud_keywords:
                score += 5.0

            importance_scores[i] = score

        return importance_scores

    def progressive_attack_single(self, text: str, max_attempts: int = 3) -> str:
        """
        对单个文本进行渐进式攻击

        Args:
            text: 原始文本
            max_attempts: 最大尝试次数

        Returns:
            对抗文本
        """
        # 原始预测
        original_pred = self.victim_model.predict([text])[0]
        # 如果原始预测已经是非欺诈，无需攻击
        if original_pred == 0:
            return text
        # 分词
        words = list(jieba.cut(text))
        if len(words) < 3:
            return text
        # 计算词的重要性
        word_importance = self._calculate_word_importance(words)
        # 按重要性排序
        sorted_indices = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
        # 尝试不同的替换比例
        replace_ratios = [0.1, 0.2, 0.3][:max_attempts]  # 10%, 20%, 30%
        for ratio in replace_ratios:
            # 生成候选对抗文本
            candidate_words = words.copy()
            n_replace = max(1, int(len(words) * ratio))
            replacements_made = 0
            for idx, _ in sorted_indices:
                if replacements_made >= n_replace:
                    break

                original_word = words[idx]
                synonyms = self._get_similar_words(original_word, top_n=2)

                if synonyms:
                    # 选择最佳同义词
                    new_word = synonyms[0]
                    candidate_words[idx] = new_word
                    replacements_made += 1
            candidate_text = ''.join(candidate_words)
            # 检查攻击是否成功
            candidate_pred = self.victim_model.predict([candidate_text])[0]
            if candidate_pred == 0:  # 攻击成功
                logger.debug(f"攻击成功，替换比例: {ratio:.0%}，替换词数: {replacements_made}")
                return candidate_text

        # 所有尝试都失败，返回原始文本
        return text

    def generate_adversarial_batch(self, texts: List[str], labels: List[int],
                                   max_attempts: int = 3) -> Tuple[List[str], float]:
        """
        批量生成对抗样本（使用渐进式攻击）

        Args:
            texts: 原始文本列表
            labels: 标签列表
            max_attempts: 最大尝试次数

        Returns:
            对抗样本列表，攻击成功率
        """
        logger.info(f"开始渐进式对抗攻击，最大尝试次数: {max_attempts}")

        adversarial_texts = []
        success_count = 0
        total_attacks = sum(1 for label in labels if label == 1)

        if total_attacks == 0:
            return texts, 0.0

        for i, (text, label) in enumerate(zip(texts, labels)):
            if i % 5 == 0 and i > 0:
                logger.info(f"处理进度: {i}/{len(texts)} (成功率: {success_count / i:.2%})")

            if label == 1:  # 只攻击欺诈样本
                adv_text = self.progressive_attack_single(text, max_attempts)

                # 检查攻击是否成功
                orig_pred = self.victim_model.predict([text])[0]
                adv_pred = self.victim_model.predict([adv_text])[0]

                if orig_pred == 1 and adv_pred == 0:
                    success_count += 1

                adversarial_texts.append(adv_text)
            else:
                adversarial_texts.append(text)

        attack_success_rate = success_count / total_attacks
        logger.info(f"渐进式攻击完成，成功率: {attack_success_rate:.2%} ({success_count}/{total_attacks})")

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

            # 攻击成功：欺诈样本被预测为非欺诈
            attack_success = sum(1 for pred in adv_fraud_preds if pred == 0)
            total_fraud = len(fraud_indices)
            attack_success_rate = attack_success / total_fraud if total_fraud > 0 else 0
        else:
            attack_success_rate = 0

        # 计算文本相似度
        similarities = []
        for i, (orig, adv) in enumerate(zip(original_texts, adversarial_texts)):
            if labels[i] == 1:  # 只计算欺诈样本
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
