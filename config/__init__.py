# config/__init__.py
import yaml
from pathlib import Path
import os


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
    config['TRAIN_CLEANED'] = str(project_root / config['data']['train_cleaned'])
    config['TEST_CLEANED'] = str(project_root / config['data']['test_cleaned'])
    config['PROCESSED'] = str(project_root / config['data']['processed'])

    # 模型参数
    config['MAX_LENGTH'] = config['model']['max_length']
    config['BATCH_SIZE'] = config['model']['batch_size']
    config['LEARNING_RATE'] = config['model']['learning_rate']
    config['NUM_EPOCHS'] = config['model']['num_epochs']
    config['HIDDEN_DROPOUT_PROB'] = config['model']['hidden_dropout_prob']

    # 攻击参数
    config['MAX_ITERATIONS'] = config['attack']['max_iterations']
    config['BEAM_SIZE'] = config['attack']['beam_size']
    config['MAX_CANDIDATES'] = config['attack']['max_candidates']
    config['SIMILARITY_THRESHOLD'] = config['attack']['similarity_threshold']

    return config


# 创建全局配置对象
CONFIG = get_config()