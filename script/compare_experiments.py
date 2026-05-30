# compare_experiments.py
import torch
from train import main as train_main
from evaluate import evaluate
from utils import set_seed, Timer
from config import *
import pandas as pd

def get_model_size(backbone):
    from models import FCN8s
    model = FCN8s(backbone=backbone, num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    # 模型文件大小
    temp_path = os.path.join(SAVE_DIR, f'{backbone}_temp.pth')
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return total_params / 1e6, size_mb  # 百万参数，MB

def main():
    results = []
    backbones = ['resnet18', 'resnet34']
    for bk in backbones:
        print(f"\n=== Experiment with backbone: {bk} ===")
        set_seed(SEED)
        # 训练
        timer = Timer()
        timer.start()
        train_main(bk)
        train_time = timer.stop()
        # 评估
        miou, pa = evaluate(bk)
        param_m, size_mb = get_model_size(bk)
        results.append({
            'Backbone': bk,
            'mIoU': f'{miou:.4f}',
            'Pixel Accuracy': f'{pa:.4f}',
            'Training Time (s)': f'{train_time:.2f}',
            'Params (M)': f'{param_m:.2f}',
            'Model Size (MB)': f'{size_mb:.2f}'
        })
    df = pd.DataFrame(results)
    print("\n=== Comparison Results ===")
    print(df.to_string(index=False))
    df.to_csv('comparison_results.csv', index=False)

if __name__ == '__main__':
    main()
