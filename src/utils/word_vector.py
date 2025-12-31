# src/utils/word_vector.py
import numpy as np
import pickle
import os
from tqdm import tqdm
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordVectorManager:
    """词向量管理器"""

    def __init__(self, vector_path: str = None, cache_path: str = None):
        self.vectors: Dict[str, np.ndarray] = {}
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}

        if cache_path and os.path.exists(cache_path):
            self.load_cache(cache_path)
        elif vector_path:
            self.load_vectors(vector_path)

    def load_vectors(self, vector_path: str, limit: int = 200000):
        """加载词向量文件"""
        logger.info(f"正在加载词向量: {vector_path}")

        try:
            with open(vector_path, 'r', encoding='utf-8') as f:
                # 读取第一行获取词向量信息
                line = f.readline().strip()
                if ' ' in line:
                    vocab_size, dim = map(int, line.split())
                    logger.info(f"词向量信息: 词汇量={vocab_size}, 维度={dim}")
                else:
                    # 有些文件第一行不是元信息
                    f.seek(0)
                    vocab_size, dim = limit, 300  # 默认值

                count = 0
                for line in tqdm(f, total=vocab_size, desc="加载词向量"):
                    if count >= limit:
                        break

                    parts = line.rstrip().split(' ')
                    word = parts[0]

                    try:
                        vector = np.array([float(x) for x in parts[1:]])
                        self.vectors[word] = vector
                        self.word_index[word] = count
                        self.index_word[count] = word
                        count += 1
                    except:
                        continue

            logger.info(f"成功加载 {len(self.vectors)} 个词向量")

            # 构建词向量矩阵用于快速计算
            self._build_vector_matrix()

        except Exception as e:
            logger.error(f"加载词向量失败: {e}")
            # 使用一个小的示例词向量
            self._load_example_vectors()

    def _load_example_vectors(self):
        """加载示例词向量（用于测试）"""
        logger.info("使用示例词向量")
        # 构建一个小的词向量表
        words = ['贷款', '客服', '银行', '诈骗', '退款', '账户', '密码', '链接']
        dim = 300

        for i, word in enumerate(words):
            self.vectors[word] = np.random.randn(dim)
            self.word_index[word] = i
            self.index_word[i] = word

        self._build_vector_matrix()

    def _build_vector_matrix(self):
        """构建词向量矩阵"""
        if not self.vectors:
            return

        words = list(self.vectors.keys())
        dim = len(next(iter(self.vectors.values())))

        self.vector_matrix = np.zeros((len(words), dim))
        for i, word in enumerate(words):
            self.vector_matrix[i] = self.vectors[word]

        # 归一化用于余弦相似度计算
        self.norm_matrix = self.vector_matrix / np.linalg.norm(self.vector_matrix, axis=1, keepdims=True)

    def get_vector(self, word: str) -> np.ndarray:
        """获取词向量"""
        return self.vectors.get(word, None)

    def get_similar_words(self, word: str, top_n: int = 10, min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """获取相似词（余弦相似度）"""
        if word not in self.vectors:
            return []

        word_vec = self.get_vector(word)
        if word_vec is None:
            return []

        # 归一化查询向量
        norm_word_vec = word_vec / np.linalg.norm(word_vec)

        # 计算余弦相似度
        similarities = np.dot(self.norm_matrix, norm_word_vec)

        # 获取最相似的前top_n个词（排除自身）
        similar_indices = np.argsort(similarities)[::-1]
        similar_words = []

        for idx in similar_indices:
            if len(similar_words) >= top_n + 1:  # +1 为了排除自己
                break

            similar_word = self.index_word[idx]
            similarity = similarities[idx]

            if similar_word == word:
                continue

            if similarity < min_similarity:
                break

            similar_words.append((similar_word, float(similarity)))

        return similar_words[:top_n]

    def find_synonyms(self, word: str, top_n: int = 5) -> List[str]:
        """查找同义词"""
        similar_words = self.get_similar_words(word, top_n)
        return [word for word, score in similar_words]

    def save_cache(self, cache_path: str):
        """保存缓存"""
        cache_data = {
            'vectors': self.vectors,
            'word_index': self.word_index,
            'index_word': self.index_word
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"词向量缓存已保存: {cache_path}")

    def load_cache(self, cache_path: str):
        """加载缓存"""
        logger.info(f"正在加载词向量缓存: {cache_path}")

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.vectors = cache_data['vectors']
            self.word_index = cache_data['word_index']
            self.index_word = cache_data['index_word']

            self._build_vector_matrix()
            logger.info(f"成功加载 {len(self.vectors)} 个词向量")

        except Exception as e:
            logger.error(f"加载缓存失败: {e}")

    def batch_find_synonyms(self, words: List[str], top_n: int = 5) -> Dict[str, List[str]]:
        """批量查找同义词"""
        synonyms_dict = {}

        for word in words:
            synonyms = self.find_synonyms(word, top_n)
            if synonyms:
                synonyms_dict[word] = synonyms

        return synonyms_dict


# 全局词向量管理器实例
_word_vector_manager = None


def get_word_vector_manager(vector_path: str = None, cache_path: str = None):
    """获取全局词向量管理器"""
    global _word_vector_manager

    if _word_vector_manager is None:
        if cache_path and os.path.exists(cache_path):
            _word_vector_manager = WordVectorManager(cache_path=cache_path)
        elif vector_path:
            _word_vector_manager = WordVectorManager(vector_path=vector_path)
        else:
            # 默认路径
            default_path = "data/word_vectors/cc.zh.300.vec"
            if os.path.exists(default_path):
                _word_vector_manager = WordVectorManager(vector_path=default_path)
            else:
                logger.warning("未找到词向量文件，使用示例词向量")
                _word_vector_manager = WordVectorManager()

    return _word_vector_manager