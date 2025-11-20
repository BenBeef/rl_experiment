import torch

x = torch.tensor([0.5, 1.5, 2.5, 3.5])
y = torch.clamp(x, min=1.0, max=3.0)
print(y)  # 输出: tensor([1.0, 1.5, 2.5, 3.0])