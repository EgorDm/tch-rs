import tchpy
import torch

eye = torch.eye(3)
eye.grad = torch.ones(3, 3)
print('aaaaa')
eye2 = tchpy.print(eye)
print(eye)
print('eye2')
print(eye2)