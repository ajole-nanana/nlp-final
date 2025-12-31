# src/models/svm_model.py
import joblib
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from config import CONFIG


class SVMModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,  # 中文需要自定义停用词
            min_df=3
        )
        self.model = SVC(
            kernel='linear',  # 对于文本分类，线性SVM通常表现较好
            C=1.0,
            probability=True,
            random_state=42
        )

    def train(self, x_train, y_train):
        """训练SVM模型"""
        print("正在提取TF-IDF特征...")
        x_train_tfidf = self.vectorizer.fit_transform(x_train)
        print(f"特征维度: {x_train_tfidf.shape}")

        print("正在训练SVM模型...")
        self.model.fit(x_train_tfidf, y_train)
        print("训练完成!")

    def predict(self, x):
        """预测"""
        x_tfidf = self.vectorizer.transform(x)
        return self.model.predict(x_tfidf)

    def predict_proba(self, x):
        """预测概率"""
        x_tfidf = self.vectorizer.transform(x)
        return self.model.predict_proba(x_tfidf)

    def evaluate(self, x_test, y_test):
        """评估模型"""
        y_pred = self.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"测试集准确率: {acc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        return acc

    def save(self, path):
        """保存模型"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model
        }, path)
        print(f"模型已保存到: {path}")

    @staticmethod
    def load(path):
        """加载模型"""
        data = joblib.load(path)
        svm_model = SVMModel()
        svm_model.vectorizer = data['vectorizer']
        svm_model.model = data['model']
        return svm_model


def train_svm_model():
    """训练SVM受害模型的主函数"""
    print("=== 训练SVM受害模型 ===")

    # 加载清洗后的数据
    try:
        train_df = pd.read_csv(CONFIG['TRAIN_CLEANED'])
        test_df = pd.read_csv(CONFIG['TEST_CLEANED'])

        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")

        # 准备数据
        x_train = train_df['specific_dialogue_content'].astype(str).tolist()
        y_train = train_df['is_fraud'].astype(int).tolist()
        x_test = test_df['specific_dialogue_content'].astype(str).tolist()
        y_test = test_df['is_fraud'].astype(int).tolist()

        # 训练模型
        svm = SVMModel()
        svm.train(x_train, y_train)

        # 评估模型
        print("\n=== 模型评估 ===")
        svm.evaluate(x_test, y_test)

        # 保存模型
        model_path = CONFIG['ROOT_DIR'] + "/experiments/save_models/svm_model.pkl"
        svm.save(model_path)

        return svm

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return None


if __name__ == "__main__":
    train_svm_model()
