import torch
import torchvision.models as models
import random

checkpoint = torch.load('citeseer-knowledge.pth.tar')	# 加载模型
print(checkpoint.keys())			#dict_keys(['logits', 'embedding'])									# 查看模型元素

'''
print(checkpoint['logits'])
print(checkpoint['logits'].shape)   #torch.Size([2708, 7])
print(checkpoint['embedding'])
print(checkpoint['embedding'].shape)    #torch.Size([2708, 64])
'''
seed = random.randint(0, 3000)
print("seed: ", seed)