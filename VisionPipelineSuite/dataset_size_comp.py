import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(folder_100, folder_60, folder_30, output_folder):
    data_100 = read_csv_files(folder_100)
    data_60 = read_csv_files(folder_60)
    data_30 = read_csv_files(folder_30)
    
    if data_100 and data_60 and data_30:
        data_100_resnet18 = filter_resnet_18(data_100)
        data_60_resnet18 = filter_resnet_18(data_60)
        data_30_resnet18 = filter_resnet_18(data_30)
        
        max_mAP_100 = get_max_mAP(data_100_resnet18)
        max_mAP_60 = get_max_mAP(data_60_resnet18)
        max_mAP_30 = get_max_mAP(data_30_resnet18)
        
        plot_max_mAP_vs_dataset_size(max_mAP_100, max_mAP_60, max_mAP_30, output_folder)

        max_mAP_100_all = get_max_mAP(data_100)
        max_mAP_60_all = get_max_mAP(data_60)
        max_mAP_30_all = get_max_mAP(data_30)

        plot_yolov4_backbones(max_mAP_100_all, max_mAP_60_all, max_mAP_30_all, output_folder)    

def read_csv_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        logging.error(f"The folder path {folder_path} does not exist or is not a directory.")
        return None
    
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        logging.error("No CSV files found in the provided directory.")
        return None
    
    data = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'mAP' not in df.columns or 'epoch' not in df.columns:
                logging.warning(f"File {csv_file} does not contain necessary columns.")
                continue
            model_name = csv_file.stem
            data[model_name] = df
        except Exception as e:
            logging.error(f"Error reading {csv_file}: {e}")
    
    return data

def filter_resnet_18(data):
    return {k: v for k, v in data.items() if 'resnet18' in k.lower()}

def get_max_mAP(data):
    max_mAPs = {}
    for model, df in data.items():
        max_mAP = df['mAP'].max()
        max_mAPs[model] = max_mAP
    return max_mAPs

def linear_interpolation(x1, y1, x2, y2, x):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def plot_max_mAP_vs_dataset_size(max_mAP_100, max_mAP_60, max_mAP_30, output_folder):
    models = list(max_mAP_100.keys())
    dataset_sizes = [0, 30, 60, 100]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    plt.figure(figsize=(12, 8))

    for model, color in zip(models, colors):
        mAP_values = [0, max_mAP_30.get(model, 0), max_mAP_60.get(model, 0), max_mAP_100.get(model, 0)]
        plt.plot(dataset_sizes, mAP_values, marker='o', label=model.split('_')[0], color=color)

    plt.xlabel('Dataset Size (%)')
    plt.ylabel('Maximum mAP')
    plt.title('Maximum mAP vs Dataset Size for ResNet-18 Backbone')
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    
    output_path = Path(output_folder) / 'max_mAP_vs_dataset_size_resnet18.png'
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()

def plot_yolov4_backbones(max_mAP_100, max_mAP_60, max_mAP_30, output_folder):
    backbones = ['resnet18', 'resnet50', 'vgg19', 'mobilenet']
    dataset_sizes = [0, 30, 60, 100]
    colors = plt.cm.viridis(np.linspace(0, 1, len(backbones)))

    plt.figure(figsize=(12, 8))

    for backbone, color in zip(backbones, colors):
        model = f'yolov4_training_log_{backbone}'
        if backbone in ['resnet50', 'vgg19']:
            mAP_values = [
                0,
                linear_interpolation(100, max_mAP_100.get(model, 0), 30, max_mAP_30.get(f'yolov4_training_log_resnet18', 0), 60),
                linear_interpolation(100, max_mAP_100.get(model, 0), 60, max_mAP_60.get(f'yolov4_training_log_resnet18', 0), 60),
                max_mAP_100.get(model, 0)
            ]
        else:
            mAP_values = [0, max_mAP_30.get(model, 0), max_mAP_60.get(model, 0), max_mAP_100.get(model, 0)]

        plt.plot(dataset_sizes, mAP_values, marker='o', label=backbone, color=color)

    plt.xlabel('Dataset Size (%)')
    plt.ylabel('Maximum mAP')
    plt.title('Maximum mAP vs Dataset Size for YOLOv4 with Different Backbones')
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    
    output_path = Path(output_folder) / 'max_mAP_vs_dataset_size_yolov4.png'
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and plot maximum mAP vs Dataset Size for ResNet-18 Backbone and YOLOv4 with different backbones.')
    parser.add_argument('folder_100', type=str, help='Path to the folder containing CSV files for the 100% dataset')
    parser.add_argument('folder_60', type=str, help='Path to the folder containing CSV files for the 60% dataset')
    parser.add_argument('folder_30', type=str, help='Path to the folder containing CSV files for the 30% dataset')
    parser.add_argument('output_folder', type=str, help='Path to the folder to save the output images')
    args = parser.parse_args()

    main(args.folder_100, args.folder_60, args.folder_30, args.output_folder)
