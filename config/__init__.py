# config/__init__.py
import yaml
from pathlib import Path
import os


def get_config():
    """获取全局配置"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # 路径解析 - 将相对路径转换为绝对路径
    config['ROOT_DIR'] = str(project_root)

    # 数据路径
    config['TRAIN_DATA'] = str(project_root / config['data']['train_data'])
    config['TEST_DATA'] = str(project_root / config['data']['test_data'])
    config['TRAIN_CLEANED'] = str(project_root / config['data']['train_cleaned'])
    config['TEST_CLEANED'] = str(project_root / config['data']['test_cleaned'])
    config['PROCESSED'] = str(project_root / config['data']['processed'])

    return config


# 创建全局配置对象
CONFIG = get_config()