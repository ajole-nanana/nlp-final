# nlp-final
project for nlp final work

```
fraudslow_project/
│
├── README.md          # 项目说明，如何安装和运行
├── requirements.txt    # Python依赖列表
├── config.yaml         # 配置文件，统一管理超参数
│
├── data/               # 数据目录
│   ├── raw/           # 存放原始数据集
│   ├── processed/     # 存放预处理后的数据
│   └── adversarial/   # 存放生成的对抗样本
│
├── src/                # 源代码目录
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models/         # 模型定义和加载
│   ├── attackers/      # 攻击器核心代码
│   └── utils/          # 工具函数（如相似度计算）
│
└── experiments/        # 实验脚本和结果输出
    ├── train_baseline.py
    ├── run_attack.py
    └── results/        # 自动保存实验结果和图表
```

