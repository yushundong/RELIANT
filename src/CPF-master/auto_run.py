import os

dataset = ["bail"] # bail\credit\raw
teacher = ["GCN", "GraphSAGE"] # GCN\GraphSAGE\SGC\APPNP\GAT

for d in dataset:
    for t in teacher:
        for i in range(10):
            if i == 0 :
                os.system(f"python train_dgl.py --dataset={d} --teacher={t}")
            os.system(f"python spawn_worker.py --dataset={d} --teacher={t}")
    print(f"{d} {t}: done!")

