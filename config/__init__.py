import yaml
from pathlib import Path


def get_config():
    """获取全局配置"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # 路径解析
    config['ROOT_DIR'] = str(project_root)
    config['TRAIN_DATA'] = str(project_root / config['data']['train_data'])
    config['TEST_DATA'] = str(project_root / config['data']['test_data'])
    config['PROCESSED'] = str(project_root / config['data']['processed'])

    return config


# 创建全局配置对象
CONFIG = get_config()
