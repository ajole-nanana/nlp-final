# experiments/train_victim.py
import sys
import os
from src.models.svm_model import train_svm_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    print("开始训练受害模型...")
    model = train_svm_model()
    if model:
        print("\n受害模型训练完成!")
        # 可以添加更多评估或可视化代码
    else:
        print("训练失败!")
