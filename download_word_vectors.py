# scripts/download_word_vectors.py
import os
import requests
import gzip
import shutil
from tqdm import tqdm


def download_wiki_vectors():
    """下载中文维基百科词向量"""
    print("正在下载中文维基百科词向量...")

    # 较小的词向量文件（推荐用于实验）
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz"
    save_path = "data/word_vectors/cc.zh.300.vec.gz"
    extracted_path = "data/word_vectors/cc.zh.300.vec"

    # 创建目录
    os.makedirs("data/word_vectors", exist_ok=True)

    # 下载文件
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载中") as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))

    print("下载完成！正在解压...")

    # 解压文件
    with gzip.open(save_path, 'rb') as f_in:
        with open(extracted_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"词向量已保存到: {extracted_path}")
    return extracted_path


if __name__ == "__main__":
    download_wiki_vectors()
