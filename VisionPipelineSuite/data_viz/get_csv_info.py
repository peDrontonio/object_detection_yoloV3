import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(folder_path, ouput_folder, gamma, generate_latex):

    data = read_csv_files(folder_path)
    if data:
        plot_mAP_vs_epochs(data, output_folder, gamma)
        plot_mAP_per_detector(data, output_folder, gamma)

        if generate_latex:
            generate_latex_table(data)

def parse_file_name(file_name):
    parts = file_name.stem.split('_')
    if len(parts) >= 4:
        object_detector = parts[0]
        backbone = parts[-1]
        return f"{object_detector} + {backbone}"
    return file_name.stem

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
            model_name = parse_file_name(csv_file)
            data[model_name] = df
        except Exception as e:
            logging.error(f"Error reading {csv_file}: {e}")
    
    return data

def exponential_color_mapping(value, color, gamma):
    value = np.clip(value, 0, 1) ** gamma
    gray = np.array([0.5, 0.5, 0.5])
    target_color = np.array(color)
    return gray * (1 - value) + target_color * value

def rank_detectors_by_mAP(data):
    detectors = list(data.keys())
    max_mAPs = []
    for detector in detectors:
        df = data[detector]
        df = df.dropna(subset=['mAP'])
        max_mAPs.append(df['mAP'].max())
    
    # Sort detectors by max mAP
    sorted_detectors = [detector for _, detector in sorted(zip(max_mAPs, detectors), reverse=True)]
    return sorted_detectors

def plot_mAP_vs_epochs(data, output_folder, gamma):
    plt.figure(figsize=(12, 8))

    forest_green = [0.13, 0.55, 0.13]

    sorted_detectors = rank_detectors_by_mAP(data)
    max_mAPs = [data[detector].dropna(subset=['mAP'])['mAP'].max() for detector in sorted_detectors]

    # Normalize and map colors
    max_mAPs = np.array(max_mAPs)
    norm_max_mAPs = (max_mAPs - max_mAPs.min()) / (max_mAPs.max() - max_mAPs.min())
    colors = [exponential_color_mapping(value, forest_green, gamma) for value in norm_max_mAPs]

    color_map = dict(zip(sorted_detectors, colors))

    for detector in sorted_detectors:
        df = data[detector]
        df = df.dropna(subset=['mAP'])
        if 0 not in df['epoch'].values:
            df = pd.concat([pd.DataFrame({'epoch': [0], 'mAP': [0]}), df], ignore_index=True)
        df = df[df['epoch'] <= 80]
        epochs = df['epoch']
        mAPs = df['mAP']

        color = color_map[detector]
        plt.plot(epochs, mAPs, color=color, label=detector)

    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision (mAP) over Epochs of All Model Combinations')
    plt.xlim(0, 80)
    plt.ylim(0, 1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    
    output_path = Path(output_folder) / 'mAP_vs_epochs.png'
    plt.savefig(output_path, bbox_inches='tight')

def plot_mAP_per_detector(data, output_folder, gamma):
    detector_types = list(set(detector.split(' + ')[0] for detector in data.keys()))
    colors = plt.cm.viridis(np.linspace(0, 1, len(detector_types)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()

    for ax, (detector_type, color) in zip(axes, zip(detector_types, colors)):
        detector_data = {k: v for k, v in data.items() if k.startswith(detector_type)}

        sorted_detectors = rank_detectors_by_mAP(detector_data)
        max_mAPs = [detector_data[detector].dropna(subset=['mAP'])['mAP'].max() for detector in sorted_detectors]

        max_mAPs = np.array(max_mAPs)
        norm_max_mAPs = (max_mAPs - max_mAPs.min()) / (max_mAPs.max() - max_mAPs.min())
        individual_colors = [exponential_color_mapping(value, color[:3], gamma) for value in norm_max_mAPs]

        color_map = dict(zip(sorted_detectors, individual_colors))

        for detector in sorted_detectors:
            df = detector_data[detector]
            df = df.dropna(subset=['mAP'])
            if 0 not in df['epoch'].values:
                df = pd.concat([pd.DataFrame({'epoch': [0], 'mAP': [0]}), df], ignore_index=True)
            df = df[df['epoch'] <= 80]
            epochs = df['epoch']
            mAPs = df['mAP']

            color = color_map[detector]
            ax.plot(epochs, mAPs, color=color, label=detector)

        ax.set_title(f'{detector_type.upper()}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left')
        ax.grid(True)

        # Save individual plots
        plt.figure(figsize=(12, 8))
        for detector in sorted_detectors:
            df = detector_data[detector]
            df = df.dropna(subset=['mAP'])
            if 0 not in df['epoch'].values:
                df = pd.concat([pd.DataFrame({'epoch': [0], 'mAP': [0]}), df], ignore_index=True)
            df = df[df['epoch'] <= 80]
            epochs = df['epoch']
            mAPs = df['mAP']

            color = color_map[detector]
            plt.plot(epochs, mAPs, color=color, label=detector)

        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title(f'Mean Average Precision (mAP) over Epochs for {detector_type.upper()}', fontweight='bold')
        plt.xlim(0, 80)
        plt.ylim(0, 1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)

        output_path = Path(output_folder) / f'mAP_vs_epochs_{detector_type}.png'
        plt.savefig(output_path, bbox_inches='tight')

    plt.tight_layout()
    output_path = Path(output_folder) / 'mAP_2x2_grid.png'
    fig.savefig(output_path, bbox_inches='tight')

def generate_latex_table(data):
    models = ['Faster-RCNN', 'SSD', 'DSSD', 'YOLOv4', 'RetinaNet']
    backbones = ['ResNet18', 'ResNet50', 'VGG19', 'MobileNet']
    
    latex_table = r"""
        \begin{table}[ht]
            \centering
            \caption{FPS Performance on a Jetson Orin AGX}
            \label{tab:fps_results}
            \begin{tabular}{lcccc}
                \toprule
                Model & ResNet-18 & ResNet-50 & VGG-19 & MobileNet \\
                \midrule
    """
    
    for model in models:
        row = f"        {model} "
        for backbone in backbones:
            key = f"{model.lower()} + {backbone.lower()}"
            if key in data:

                df = data[key]
                fps_value = df['mAP'].max()  # Replace with appropriate logic to get the FPS value
                row += f"& {np.round(fps_value, 2)} "

            else:
                row += "& * "  # Placeholder for missing values

        row += r"\\"
        latex_table += f"{row}\n"
    
    latex_table += r"""
                \bottomrule
            \end{tabular}
            \smallskip
            \footnotesize{Source: Own author.}
        \end{table}
    """
    
    print(latex_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot mAP vs Epochs for different object detectors.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing CSV files')
    parser.add_argument('output_folder', type=str, help='Path to the folder to save the output images')
    parser.add_argument('--gamma', type=float, default=2.2, help='Gamma value for the exponential color mapping')
    parser.add_argument('--latex', action='store_true', help='Generate a LaTeX table for FPS performance')

    args = parser.parse_args()
    
    main(**vars(args))

    
    """
    Example usage:
    main('path/to/csv/folder', 'path/to/output/folder', gamma=2.2, generate_latex=True)
    
    - The script reads CSV files from the specified folder.
    - It generates plots for mAP vs epochs and mAP per detector.
    - It can also generate a LaTeX table for FPS performance.
    - The gamma value is used for color mapping in the plots.
    - The --latex flag generates a LaTeX table for FPS performance.

    """