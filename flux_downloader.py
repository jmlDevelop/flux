import os
import argparse
from huggingface_hub import hf_hub_download
from typing import List, Tuple

# 从 util.py 导入必要的配置
from flux.util import configs

def download_model(model_name: str, output_dir: str) -> List[Tuple[str, str]]:
    """
    下载指定的模型文件到指定目录。

    :param model_name: 模型名称（例如 'flux-schnell'）
    :param output_dir: 下载文件保存的目录
    :return: 下载文件的列表，每个元素是一个元组 (文件名, 保存路径)
    """
    config = configs[model_name]
    downloaded_files = []

    # 下载 Flux 模型
    if config.repo_id and config.repo_flow:
        flux_path = hf_hub_download(config.repo_id, config.repo_flow, cache_dir=output_dir)
        downloaded_files.append((config.repo_flow, flux_path))

    # 下载自动编码器
    if config.repo_id and config.repo_ae:
        ae_path = hf_hub_download(config.repo_id, config.repo_ae, cache_dir=output_dir)
        downloaded_files.append((config.repo_ae, ae_path))

    return downloaded_files

def main():
    parser = argparse.ArgumentParser(description="Download Flux models to a specified directory.")
    parser.add_argument("--model", choices=list(configs.keys()), required=True, help="Model to download")
    parser.add_argument("--output", required=True, help="Directory to save downloaded files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Downloading {args.model} to {args.output}")
    downloaded_files = download_model(args.model, args.output)

    print("\nDownloaded files:")
    for file_name, file_path in downloaded_files:
        print(f"- {file_name}: {file_path}")

    print("\nTo use these local files, set the following environment variables:")
    if args.model == "flux-schnell":
        print(f"export FLUX_SCHNELL={[path for name, path in downloaded_files if 'flux1-schnell' in name][0]}")
    elif args.model == "flux-dev":
        print(f"export FLUX_DEV={[path for name, path in downloaded_files if 'flux1-dev' in name][0]}")
    print(f"export AE={[path for name, path in downloaded_files if 'ae.safetensors' in name][0]}")

if __name__ == "__main__":
    main()