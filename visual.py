import torch

# 加载模型的checkpoint
checkpoint_path = '/home/ubuntu/hjd/SparseTSF/checkpoints/ETTh1_720_720_SparseTSF_ETTh1_ftM_sl720_pl720_linear_test_0_seed2023/checkpoint.pth'
checkpoint = torch.load(checkpoint_path)

# 查看所有的参数
print("模型参数：")
for key, value in checkpoint.items():
    print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")
