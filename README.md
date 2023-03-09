# RELIANT
Open-source code for "RELIANT: Fair Knowledge Distillation for Graph Neural Networks".

## Citation

If you find **our code** and the **new datasets (folders named small and medium) for fairness research on graphs** useful, please cite our paper. Thank you!

```
@inproceedings{dong2023reliant,
  title={RELIANT: Fair Knowledge Distillation for Graph Neural Networks},
  author={Dong, Yushun and Zhang, Binchi and Yuan, Yiling and Zou, Na and Wang, Qi and Li, Jundong},
  booktitle={Proceedings of the 2023 SIAM International Conference on Data Mining},
  year={2023}
}
```

## Environment
Experiments are carried out on an Nvidia RTX A6000 with Cuda 11.1. 

Library details can be found in requirements.txt.

Notice: Cuda is enabled for default settings.

## Usage
We have three datasets for experiments, namely Recidivism, Credit, DBLP, and DBLP-L. 

First enter the directory 

```
cd CPF-master
```

## Log examples on DBLP



### 1. An example with CPF+GCN under SP

Run

```
python train_dgl.py --dataset small --teacher GCN --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:23<00:00, 43.45it/s]
Optimization Finished!
Total time elapsed: 13.8847s

| Variant              | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('small', 'GCN', 42) | 0.2497±0.0000 | 0.9222±0.0000 | 0.0676±0.0000 | 0.0182±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset small --teacher GCN --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
Loading cascades...
 62%|███████████████████████████████████████████████████████████████████████████████████████████████▍                                                           | 616/1000 [12:30<07:49,  1.22s/it]Stop!!!
 62%|███████████████████████████████████████████████████████████████████████████████████████████████▍                                                           | 616/1000 [12:31<07:48,  1.22s/it]
Optimization Finished!
Total time elapsed: 643.3283s

Final results:

| Variant              | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('small', 'PLP', 42) | 0.9303±0.0000 | 0.9292±0.0000 | 0.0217±0.0000 | 0.1029±0.0000 |
```

### 2. An example with CPF+SAGE under SP

Run

```
python train_dgl.py --dataset small --teacher GraphSAGE --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
 72%|████████████████████████████████████████████████████████████████████████████████████████████████████▍                                      | 723/1000 [00:17<00:06, 41.83it/s]Saving cascade info...
 72%|████████████████████████████████████████████████████████████████████████████████████████████████████▍                                      | 723/1000 [00:27<00:10, 26.40it/s]
Optimization Finished!
Total time elapsed: 12.5953s

| Variant                    | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|----------------------------|---------------|---------------|---------------|---------------|
| ('small', 'GraphSAGE', 42) | 0.2195±0.0000 | 0.9235±0.0000 | 0.0812±0.0000 | 0.0612±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset small --teacher GraphSAGE --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
Loading cascades...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 999/1000 [21:40<00:01,  1.25s/it]Stop!!!
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 999/1000 [21:41<00:01,  1.30s/it]
Optimization Finished!
Total time elapsed: 1109.8062s

Final results:

| Variant              | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('small', 'PLP', 42) | 0.9287±0.0000 | 0.9318±0.0000 | 0.0315±0.0000 | 0.0711±0.0000 |
```

### 3. An example with AKD+GCN under SP

Run

```
python train_dgl.py --dataset small --teacher GCN --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:24<00:00, 41.41it/s]
Optimization Finished!
Total time elapsed: 14.2494s

| Variant              | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('small', 'GCN', 42) | 0.2350±0.0000 | 0.9204±0.0000 | 0.0733±0.0000 | 0.0370±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset small --teacher GCN --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 92.036%
Teacher Parity SCORE: 0.07330734697729868
Teacher Equality SCORE: 0.03696411929170551
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [01:21<00:00,  7.38it/s]

Final results:
| Variant             | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|---------------------|---------------|---------------|---------------|---------------|
| ('small', 'GCN', 1) | 0.9189±0.0000 | 0.9228±0.0000 | 0.0581±0.0000 | 0.0030±0.0000 |
```

### 4. An example with AKD+SAGE under SP

Run

```
python train_dgl.py --dataset small --teacher GraphSAGE --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
 71%|██████████████████████████████████████████████████████████████████████████████████████████████████▋                                        | 710/1000 [00:17<00:06, 41.49it/s]Saving cascade info...
 71%|███████████████████████████████████████████████████████████████████████████████████████████████████▏                                       | 714/1000 [00:27<00:11, 25.81it/s]
Optimization Finished!
Total time elapsed: 12.5899s

| Variant                    | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|----------------------------|---------------|---------------|---------------|---------------|
| ('small', 'GraphSAGE', 42) | 0.2208±0.0000 | 0.9236±0.0000 | 0.0805±0.0000 | 0.0530±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset small --teacher GraphSAGE --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 92.361%
Teacher Parity SCORE: 0.08054204111876379
Teacher Equality SCORE: 0.05297064305684995
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [01:43<00:00,  5.81it/s]

Final results:
| Variant                   | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|---------------------------|---------------|---------------|---------------|---------------|
| ('small', 'GraphSAGE', 1) | 0.9209±0.0000 | 0.9218±0.0000 | 0.0589±0.0000 | 0.0025±0.0000 |
```

### 5. An example with CPF+GCN under EO

Run

```
python train_dgl.py --dataset small --teacher GCN --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:23<00:00, 43.45it/s]
Optimization Finished!
Total time elapsed: 13.8847s

| Variant              | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('small', 'GCN', 42) | 0.2497±0.0000 | 0.9222±0.0000 | 0.0676±0.0000 | 0.0182±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset small --teacher GCN --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
Loading cascades...
 60%|███████████████████████████████████████████████████████████████████████████▌                                                  | 600/1000 [17:52<08:39,  1.30s/it]Stop!!!
 60%|███████████████████████████████████████████████████████████████████████████▌                                                  | 600/1000 [17:53<11:55,  1.79s/it]
Optimization Finished!
Total time elapsed: 957.3622s

Final results:

| Variant              | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('small', 'PLP', 42) | 0.9329±0.0000 | 0.9331±0.0000 | 0.0766±0.0000 | 0.0345±0.0000 |
```

### 6. An example with CPF+SAGE under EO

Run

```
python train_dgl.py --dataset small --teacher GraphSAGE --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
 72%|████████████████████████████████████████████████████████████████████████████████████████████████████▍                                      | 723/1000 [00:17<00:06, 41.83it/s]Saving cascade info...
 72%|████████████████████████████████████████████████████████████████████████████████████████████████████▍                                      | 723/1000 [00:27<00:10, 26.40it/s]
Optimization Finished!
Total time elapsed: 12.5953s

| Variant                    | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|----------------------------|---------------|---------------|---------------|---------------|
| ('small', 'GraphSAGE', 42) | 0.2195±0.0000 | 0.9235±0.0000 | 0.0812±0.0000 | 0.0612±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset small --teacher GraphSAGE --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
Loading cascades...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 999/1000 [24:20<00:01,  1.25s/it]Stop!!!
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 999/1000 [24:21<00:01,  1.46s/it]
Optimization Finished!
Total time elapsed: 1271.5883s

Final results:

| Variant              | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('small', 'PLP', 42) | 0.9296±0.0000 | 0.9321±0.0000 | 0.0670±0.0000 | 0.0005±0.0000 |
```

### 7. An example with AKD+GCN under EO

Run

```
python train_dgl.py --dataset small --teacher GCN --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:24<00:00, 41.41it/s]
Optimization Finished!
Total time elapsed: 14.2494s

| Variant              | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('small', 'GCN', 42) | 0.2350±0.0000 | 0.9204±0.0000 | 0.0733±0.0000 | 0.0370±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset small --teacher GCN --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 92.036%
Teacher Parity SCORE: 0.07330734697729868
Teacher Equality SCORE: 0.03696411929170551
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [01:22<00:00,  7.29it/s]

Final results:
| Variant             | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|---------------------|---------------|---------------|---------------|---------------|
| ('small', 'GCN', 1) | 0.9173±0.0000 | 0.9225±0.0000 | 0.0695±0.0000 | 0.0370±0.0000 |
```

### 8. An example with AKD+SAGE under EO

Run

```
python train_dgl.py --dataset small --teacher GraphSAGE --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 39424 nodes.
We have 144344 edges.
 71%|██████████████████████████████████████████████████████████████████████████████████████████████████▋                                        | 710/1000 [00:17<00:06, 41.49it/s]Saving cascade info...
 71%|███████████████████████████████████████████████████████████████████████████████████████████████████▏                                       | 714/1000 [00:27<00:11, 25.81it/s]
Optimization Finished!
Total time elapsed: 12.5899s

| Variant                    | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|----------------------------|---------------|---------------|---------------|---------------|
| ('small', 'GraphSAGE', 42) | 0.2208±0.0000 | 0.9236±0.0000 | 0.0805±0.0000 | 0.0530±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset small --teacher GraphSAGE --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 92.361%
Teacher Parity SCORE: 0.08054204111876379
Teacher Equality SCORE: 0.05297064305684995
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [01:43<00:00,  5.81it/s]

Final results:
| Variant                   | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|---------------------------|---------------|---------------|---------------|---------------|
| ('small', 'GraphSAGE', 1) | 0.9226±0.0000 | 0.9231±0.0000 | 0.0775±0.0000 | 0.0485±0.0000 |
```

## Log examples on DBLP-L

### 1. An example with CPF+GCN under SP

Run

```
python train_dgl.py --dataset medium --teacher GCN --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cpu
We have 129726 nodes.
We have 1311804 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [59:33<00:00,  3.57s/it]
Optimization Finished!
Total time elapsed: 2723.4194s

| Variant               | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GCN', 42) | 0.1623±0.0000 | 0.9407±0.0000 | 0.0755±0.0000 | 0.0259±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset medium --teacher GCN --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:4
We have 129726 nodes.
We have 1311804 edges.
Loading cascades...
 94%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏       | 944/1000 [43:24<02:30,  2.68s/it]Stop!!!
 94%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏       | 944/1000 [43:26<02:34,  2.76s/it]
Optimization Finished!
Total time elapsed: 1896.9747s
Final results:
| Variant               | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('medium', 'PLP', 42) | 0.9451±0.0000 | 0.9467±0.0000 | 0.0334±0.0000 | 0.0710±0.0000 |
```

### 2. An example with CPF+SAGE under SP

Run

```
python train_dgl.py --dataset medium --teacher GraphSAGE --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cpu
We have 129726 nodes.
We have 1311804 edges.
 29%|██████████████████████████████████████▉                                                                                                | 288/1000 [1:06:53<2:48:13, 14.18s/it] 29%|███████████████████████████████████████                                                                                                | 289/1000 [1:07:07<2:48:08, 14.19s/it]
 92%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎          | 922/1000 [3:41:03<18:57, 14.58s/it]Saving cascade info...
 92%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎          | 922/1000 [3:42:18<18:48, 14.47s/it]
Optimization Finished!
Total time elapsed: 12489.5175s

| Variant                     | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GraphSAGE', 42) | 0.1618±0.0000 | 0.9409±0.0000 | 0.0789±0.0000 | 0.0372±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset medium --teacher GraphSAGE --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:4
We have 129726 nodes.
We have 1311804 edges.
Loading cascades...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 999/1000 [45:37<00:02,  2.78s/it]Stop!!!
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 999/1000 [45:39<00:02,  2.74s/it]
Optimization Finished!
Total time elapsed: 2048.4245s
Final results:
| Variant               | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('medium', 'PLP', 42) | 0.9397±0.0000 | 0.9411±0.0000 | 0.0349±0.0000 | 0.0693±0.0000 |
```

### 3. An example with AKD+GCN under SP

Run

```
python train_dgl.py --dataset medium --teacher GCN --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 129726 nodes.
We have 1311804 edges.
 91%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████             | 907/1000 [00:46<00:04, 19.77it/s]Saving cascade info...
 91%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████             | 907/1000 [01:45<00:10,  8.62it/s]
Optimization Finished!
Total time elapsed: 28.2527s

| Variant               | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GCN', 42) | 0.1646±0.0000 | 0.9410±0.0000 | 0.0680±0.0000 | 0.0135±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset medium --teacher GCN --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 94.102%
Teacher Parity SCORE: 0.06799369569420602
Teacher Equality SCORE: 0.013509012101442508
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [06:01<00:00,  1.66it/s]

Final results:
| Variant              | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GCN', 1) | 0.9333±0.0000 | 0.9360±0.0000 | 0.0369±0.0000 | 0.0647±0.0000 |
```

### 4. An example with AKD+SAGE under SP

Run

```
python train_dgl.py --dataset medium --teacher GraphSAGE --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 129726 nodes.
We have 1311804 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:59<00:00, 16.70it/s]
Optimization Finished!
Total time elapsed: 42.9725s

| Variant                     | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GraphSAGE', 42) | 0.1623±0.0000 | 0.9406±0.0000 | 0.0747±0.0000 | 0.0207±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset medium --teacher GraphSAGE --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 94.065%
Teacher Parity SCORE: 0.074683525020946
Teacher Equality SCORE: 0.02070731895567679
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [08:09<00:00,  1.23it/s]

Final results:
| Variant                    | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GraphSAGE', 1) | 0.9363±0.0000 | 0.9389±0.0000 | 0.0320±0.0000 | 0.0805±0.0000 |
```

### 5. An example with CPF+GCN under EO

Run

```
python train_dgl.py --dataset medium --teacher GCN --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cpu
We have 129726 nodes.
We have 1311804 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [59:33<00:00,  3.57s/it]
Optimization Finished!
Total time elapsed: 2723.4194s

| Variant               | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GCN', 42) | 0.1623±0.0000 | 0.9407±0.0000 | 0.0755±0.0000 | 0.0259±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset medium --teacher GCN --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:4
We have 129726 nodes.
We have 1311804 edges.
Loading cascades...
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                           | 805/1000 [36:49<08:56,  2.75s/it]Stop!!!
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                           | 805/1000 [36:52<08:55,  2.75s/it]
Optimization Finished!
Total time elapsed: 1671.9693s
Final results:
| Variant               | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('medium', 'PLP', 42) | 0.9470±0.0000 | 0.9476±0.0000 | 0.0672±0.0000 | 0.0221±0.0000 |
```

### 6. An example with CPF+SAGE under EO

Run

```
python train_dgl.py --dataset medium --teacher GraphSAGE --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cpu
We have 129726 nodes.
We have 1311804 edges.
 29%|██████████████████████████████████████▉                                                                                                | 288/1000 [1:06:53<2:48:13, 14.18s/it] 29%|███████████████████████████████████████                                                                                                | 289/1000 [1:07:07<2:48:08, 14.19s/it]
 92%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎          | 922/1000 [3:41:03<18:57, 14.58s/it]Saving cascade info...
 92%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎          | 922/1000 [3:42:18<18:48, 14.47s/it]
Optimization Finished!
Total time elapsed: 12489.5175s

| Variant                     | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GraphSAGE', 42) | 0.1618±0.0000 | 0.9409±0.0000 | 0.0789±0.0000 | 0.0372±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset medium --teacher GraphSAGE --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:4
We have 129726 nodes.
We have 1311804 edges.
Loading cascades...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 999/1000 [45:37<00:02,  2.76s/it]Stop!!!
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 999/1000 [45:39<00:02,  2.74s/it]
Optimization Finished!
Total time elapsed: 2039.4652s
Final results:
| Variant               | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('medium', 'PLP', 42) | 0.9429±0.0000 | 0.9432±0.0000 | 0.0724±0.0000 | 0.0180±0.0000 |
```

### 7. An example with AKD+GCN under EO

Run

```
python train_dgl.py --dataset medium --teacher GCN --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 129726 nodes.
We have 1311804 edges.
 91%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████             | 907/1000 [00:46<00:04, 19.77it/s]Saving cascade info...
 91%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████             | 907/1000 [01:45<00:10,  8.62it/s]
Optimization Finished!
Total time elapsed: 28.2527s

| Variant               | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GCN', 42) | 0.1646±0.0000 | 0.9410±0.0000 | 0.0680±0.0000 | 0.0135±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset medium --teacher GCN --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 94.102%
Teacher Parity SCORE: 0.06799369569420602
Teacher Equality SCORE: 0.013509012101442508
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [06:06<00:00,  1.64it/s]

Final results:
| Variant              | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GCN', 1) | 0.9355±0.0000 | 0.9377±0.0000 | 0.0640±0.0000 | 0.0037±0.0000 |
```

### 8. An example with AKD+SAGE under EO

Run

```
python train_dgl.py --dataset medium --teacher GraphSAGE --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 129726 nodes.
We have 1311804 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:59<00:00, 16.70it/s]
Optimization Finished!
Total time elapsed: 42.9725s

| Variant                     | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GraphSAGE', 42) | 0.1623±0.0000 | 0.9406±0.0000 | 0.0747±0.0000 | 0.0207±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset medium --teacher GraphSAGE --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 94.065%
Teacher Parity SCORE: 0.074683525020946
Teacher Equality SCORE: 0.02070731895567679
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [07:56<00:00,  1.26it/s]

Final results:
| Variant                    | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------------|---------------|---------------|---------------|---------------|
| ('medium', 'GraphSAGE', 1) | 0.9382±0.0000 | 0.9409±0.0000 | 0.0570±0.0000 | 0.0157±0.0000 |
```

## Log examples on Credit

### 1. An example with CPF+GCN under SP

Run

```
python train_dgl.py --dataset credit --teacher GCN --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  0
We have 30000 nodes.
We have 2873716 edges.
 50%|█████████████████████████████████████████████████████████████████████▌                                                                     | 500/1000 [00:08<00:07, 67.74it/s]Saving cascade info...
 50%|█████████████████████████████████████████████████████████████████████▉                                                                     | 503/1000 [00:08<00:08, 59.42it/s]
Optimization Finished!
Total time elapsed: 5.4535s

| Variant               | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GCN', 42) | 0.5387±0.0000 | 0.7739±0.0000 | 0.1332±0.0000 | 0.1040±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset credit --teacher GCN --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 30000 nodes.
We have 2873716 edges.
Loading cascades...
 65%|██████████████████████████████████████████████████████████████████████████████████████████▉                                                | 654/1000 [03:03<01:46,  3.24it/s]Stop!!!
 65%|██████████████████████████████████████████████████████████████████████████████████████████▉                                                | 654/1000 [03:04<01:37,  3.55it/s]
Optimization Finished!
Total time elapsed: 127.5057s

Final results:

| Variant               | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('credit', 'PLP', 42) | 0.7809±0.0000 | 0.7964±0.0000 | 0.1167±0.0000 | 0.0718±0.0000 |
```

### 2. An example with CPF+SAGE under SP

Run

```
python train_dgl.py --dataset credit --teacher GraphSAGE --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 30000 nodes.
We have 2873716 edges.
 50%|██████████████████████████████████████████████████████████████████████▏                                                                    | 505/1000 [00:29<00:27, 17.76it/s]Saving cascade info...
 51%|██████████████████████████████████████████████████████████████████████▎                                                                    | 506/1000 [00:29<00:29, 16.91it/s]
Optimization Finished!
Total time elapsed: 21.6620s

| Variant                     | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GraphSAGE', 42) | 0.5290±0.0000 | 0.7809±0.0000 | 0.1419±0.0000 | 0.1149±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset credit --teacher GraphSAGE --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 30000 nodes.
We have 2873716 edges.
Loading cascades...
 51%|██████████████████████████████████████████████████████████████████████▌                                                                    | 508/1000 [02:22<02:16,  3.62it/s]Stop!!!
 51%|██████████████████████████████████████████████████████████████████████▌                                                                    | 508/1000 [02:22<02:17,  3.57it/s]
Optimization Finished!
Total time elapsed: 98.8265s

Final results:

| Variant               | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('credit', 'PLP', 42) | 0.7795±0.0000 | 0.7823±0.0000 | 0.0252±0.0000 | 0.0168±0.0000 |
```

### 3. An example with AKD+GCN under SP

Run

```
python train_dgl.py --dataset credit --teacher GCN --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 30000 nodes.
We have 2873716 edges.
 50%|█████████████████████████████████████████████████████████████████████▋                                                                     | 501/1000 [00:15<00:10, 48.03it/s]Saving cascade info...
 50%|█████████████████████████████████████████████████████████████████████▉                                                                     | 503/1000 [00:15<00:15, 32.76it/s]
Optimization Finished!
Total time elapsed: 9.7704s

| Variant               | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GCN', 42) | 0.5387±0.0000 | 0.7739±0.0000 | 0.1332±0.0000 | 0.1040±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset credit --teacher GCN --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Namespace(dataset='credit', teacher='GCN', metric='sp', dropout=0.5, gpu=0, lr=0.01, n_epochs=600, n_hidden=64, n_layers=1, weight_decay=0.0005, alpha=10.0, self_loop=True, labelrate=20, proxy=2, seed=0, role='stu', d_critic=1, g_critic=1, n_runs=1)
_config {'dataset_source': 'npz', 'seed': 0, 'train_config': {'split': {'train_examples_per_class': 20, 'val_examples_per_class': 30}, 'standardize_graph': True}}
Device:  cuda:0
Teacher Test SCORE: 77.387%
Teacher Parity SCORE: 0.13319932602835127
Teacher Equality SCORE: 0.10404228804766014
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [01:20<00:00,  7.42it/s]

Final results:
| Variant              | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GCN', 1) | 0.7781±0.0000 | 0.7915±0.0000 | 0.0259±0.0000 | 0.0243±0.0000 |
```

### 4. An example with AKD+SAGE under SP

Run

```
python train_dgl.py --dataset credit --teacher GraphSAGE --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 30000 nodes.
We have 2873716 edges.
 50%|██████████████████████████████████████████████████████████████████████▏                                                                    | 505/1000 [00:11<00:10, 48.00it/s]Saving cascade info...
 51%|██████████████████████████████████████████████████████████████████████▎                                                                    | 506/1000 [00:11<00:11, 43.90it/s]
Optimization Finished!
Total time elapsed: 7.5217s

| Variant                     | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GraphSAGE', 42) | 0.5290±0.0000 | 0.7809±0.0000 | 0.1419±0.0000 | 0.1149±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset credit --teacher GraphSAGE --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 78.093%
Teacher Parity SCORE: 0.14190014062030198
Teacher Equality SCORE: 0.11489623517791714
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [04:59<00:00,  2.00it/s]

Final results:
| Variant                    | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GraphSAGE', 1) | 0.7807±0.0000 | 0.7909±0.0000 | 0.0116±0.0000 | 0.0095±0.0000 |
```

### 5. An example with CPF+GCN under EO

Run

```
python train_dgl.py --dataset credit --teacher GCN --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  0
We have 30000 nodes.
We have 2873716 edges.
 50%|█████████████████████████████████████████████████████████████████████▌                                                                     | 500/1000 [00:08<00:07, 67.74it/s]Saving cascade info...
 50%|█████████████████████████████████████████████████████████████████████▉                                                                     | 503/1000 [00:08<00:08, 59.42it/s]
Optimization Finished!
Total time elapsed: 5.4535s

| Variant               | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GCN', 42) | 0.5387±0.0000 | 0.7739±0.0000 | 0.1332±0.0000 | 0.1040±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset credit --teacher GCN --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 30000 nodes.
We have 2873716 edges.
Loading cascades...
 60%|████████████████████████████████████████████████████████████████████████████                                                  | 604/1000 [04:28<02:49,  2.33it/s]Stop!!!
 60%|████████████████████████████████████████████████████████████████████████████                                                  | 604/1000 [04:28<02:56,  2.25it/s]
Optimization Finished!
Total time elapsed: 210.4350s

Final results:

| Variant               | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('credit', 'PLP', 42) | 0.7797±0.0000 | 0.7963±0.0000 | 0.1275±0.0000 | 0.0783±0.0000 |
```

### 6. An example with CPF+SAGE under EO

Run

```
python train_dgl.py --dataset credit --teacher GraphSAGE --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 30000 nodes.
We have 2873716 edges.
 50%|██████████████████████████████████████████████████████████████████████▏                                                                    | 505/1000 [00:29<00:27, 17.76it/s]Saving cascade info...
 51%|██████████████████████████████████████████████████████████████████████▎                                                                    | 506/1000 [00:29<00:29, 16.91it/s]
Optimization Finished!
Total time elapsed: 21.6620s

| Variant                     | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GraphSAGE', 42) | 0.5290±0.0000 | 0.7809±0.0000 | 0.1419±0.0000 | 0.1149±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset credit --teacher GraphSAGE --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 30000 nodes.
We have 2873716 edges.
Loading cascades...
 99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 994/1000 [15:21<00:05,  1.19it/s]Stop!!!
 99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 994/1000 [15:22<00:05,  1.08it/s]
Optimization Finished!
Total time elapsed: 759.5070s

Final results:

| Variant               | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('credit', 'PLP', 42) | 0.7831±0.0000 | 0.7928±0.0000 | 0.0652±0.0000 | 0.0477±0.0000 |
```

### 7. An example with AKD+GCN under EO

Run

```
python train_dgl.py --dataset credit --teacher GCN --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 30000 nodes.
We have 2873716 edges.
 50%|█████████████████████████████████████████████████████████████████████▋                                                                     | 501/1000 [00:15<00:10, 48.03it/s]Saving cascade info...
 50%|█████████████████████████████████████████████████████████████████████▉                                                                     | 503/1000 [00:15<00:15, 32.76it/s]
Optimization Finished!
Total time elapsed: 9.7704s

| Variant               | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GCN', 42) | 0.5387±0.0000 | 0.7739±0.0000 | 0.1332±0.0000 | 0.1040±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset credit --teacher GCN --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 77.387%
Teacher Parity SCORE: 0.13319932602835127
Teacher Equality SCORE: 0.10404228804766014
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [01:33<00:00,  6.41it/s]

Final results:
| Variant              | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GCN', 1) | 0.7776±0.0000 | 0.7908±0.0000 | 0.0726±0.0000 | 0.0560±0.0000 |
```

### 8. An example with AKD+SAGE under EO

Run

```
python train_dgl.py --dataset credit --teacher GraphSAGE --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 30000 nodes.
We have 2873716 edges.
 50%|██████████████████████████████████████████████████████████████████████▏                                                                    | 505/1000 [00:11<00:10, 48.00it/s]Saving cascade info...
 51%|██████████████████████████████████████████████████████████████████████▎                                                                    | 506/1000 [00:11<00:11, 43.90it/s]
Optimization Finished!
Total time elapsed: 7.5217s

| Variant                     | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|-----------------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GraphSAGE', 42) | 0.5290±0.0000 | 0.7809±0.0000 | 0.1419±0.0000 | 0.1149±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset credit --teacher GraphSAGE --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 78.093%
Teacher Parity SCORE: 0.14190014062030198
Teacher Equality SCORE: 0.11489623517791714
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [04:58<00:00,  2.01it/s]

Final results:
| Variant                    | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|----------------------------|---------------|---------------|---------------|---------------|
| ('credit', 'GraphSAGE', 1) | 0.7792±0.0000 | 0.7893±0.0000 | 0.0753±0.0000 | 0.0565±0.0000 |
```

## Log examples on Recidivism

### 1. An example with CPF+GCN under SP

Run

```
python train_dgl.py --dataset bail --teacher GCN --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 18876 nodes.
We have 642616 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:10<00:00, 92.06it/s]
Optimization Finished!
Total time elapsed: 6.8720s

| Variant             | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|---------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GCN', 42) | 0.2577±0.0000 | 0.9225±0.0000 | 0.0619±0.0000 | 0.0225±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset bail --teacher GCN --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 18876 nodes.
We have 642616 edges.
Loading cascades...
 78%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▊                               | 776/1000 [03:14<00:55,  4.02it/s]Stop!!!
 78%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▊                               | 776/1000 [03:14<00:56,  3.99it/s]
Optimization Finished!
Total time elapsed: 131.7941s
Final results:
| Variant             | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|---------------------|---------------|---------------|---------------|---------------|
| ('bail', 'PLP', 42) | 0.8928±0.0000 | 0.8994±0.0000 | 0.0567±0.0000 | 0.0229±0.0000 |
```

### 2. An example with CPF+SAGE under SP

Run

```
python train_dgl.py --dataset bail --teacher GraphSAGE --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 18876 nodes.
We have 642616 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:38<00:00, 25.98it/s]
Optimization Finished!
Total time elapsed: 27.4833s

| Variant                   | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|---------------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GraphSAGE', 42) | 0.3154±0.0000 | 0.9145±0.0000 | 0.0697±0.0000 | 0.0285±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset bail --teacher GraphSAGE --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:3
We have 18876 nodes.
We have 642616 edges.
Loading cascades...
 51%|███████████████████████████████████████████████████████████████████████▏                                                                   | 512/1000 [02:09<02:01,  4.03it/s]Stop!!!
 51%|███████████████████████████████████████████████████████████████████████▏                                                                   | 512/1000 [02:10<02:03,  3.94it/s]
Optimization Finished!
Total time elapsed: 87.6705s
Final results:
| Variant             | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|---------------------|---------------|---------------|---------------|---------------|
| ('bail', 'PLP', 42) | 0.8910±0.0000 | 0.8986±0.0000 | 0.0552±0.0000 | 0.0173±0.0000 |
```

### 3. An example with AKD+GCN under SP

Run

```
python train_dgl.py --dataset bail --teacher GCN --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 18876 nodes.
We have 642616 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:13<00:00, 76.24it/s]
Optimization Finished!
Total time elapsed: 8.5631s

| Variant             | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|---------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GCN', 42) | 0.2594±0.0000 | 0.9231±0.0000 | 0.0618±0.0000 | 0.0193±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset bail --teacher GCN --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 92.308%
Teacher Parity SCORE: 0.06178673506462984
Teacher Equality SCORE: 0.019254548666313376
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [02:13<00:00,  4.50it/s]

Final results:
| Variant            | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|--------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GCN', 1) | 0.8899±0.0000 | 0.8951±0.0000 | 0.0553±0.0000 | 0.0277±0.0000 |
```

### 4. An example with AKD+SAGE under SP

Run

```
python train_dgl.py --dataset bail --teacher GraphSAGE --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 18876 nodes.
We have 642616 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:12<00:00, 81.56it/s]
Optimization Finished!
Total time elapsed: 8.2781s

| Variant                   | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|---------------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GraphSAGE', 42) | 0.3081±0.0000 | 0.9147±0.0000 | 0.0679±0.0000 | 0.0312±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset bail --teacher GraphSAGE --metric sp
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 91.471%
Teacher Parity SCORE: 0.06787802435114115
Teacher Equality SCORE: 0.031160572337042947
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [02:18<00:00,  4.34it/s]

Final results:
| Variant                  | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|--------------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GraphSAGE', 1) | 0.8978±0.0000 | 0.9012±0.0000 | 0.0649±0.0000 | 0.0350±0.0000 |
```

### 5. An example with CPF+GCN under EO

Run

```
python train_dgl.py --dataset bail --teacher GCN --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 18876 nodes.
We have 642616 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:10<00:00, 92.06it/s]
Optimization Finished!
Total time elapsed: 6.8720s

| Variant             | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|---------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GCN', 42) | 0.2577±0.0000 | 0.9225±0.0000 | 0.0619±0.0000 | 0.0225±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset bail --teacher GCN --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 18876 nodes.
We have 642616 edges.
Loading cascades...
 55%|████████████████████████████████████████████████████████████████████████████▏                                                              | 548/1000 [02:17<01:52,  4.03it/s]Stop!!!
 55%|████████████████████████████████████████████████████████████████████████████▏                                                              | 548/1000 [02:18<01:53,  3.97it/s]
Optimization Finished!
Total time elapsed: 92.3854s
Final results:
| Variant             | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|---------------------|---------------|---------------|---------------|---------------|
| ('bail', 'PLP', 42) | 0.8663±0.0000 | 0.8710±0.0000 | 0.0442±0.0000 | 0.0072±0.0000 |
```

### 6. An example with CPF+SAGE under EO

Run

```
python train_dgl.py --dataset bail --teacher GraphSAGE --framework CPF
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 18876 nodes.
We have 642616 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:38<00:00, 25.98it/s]
Optimization Finished!
Total time elapsed: 27.4833s

| Variant                   | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|---------------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GraphSAGE', 42) | 0.3154±0.0000 | 0.9145±0.0000 | 0.0697±0.0000 | 0.0285±0.0000 |
```

After obtaining the teacher knowledge, then run

```
python spawn_worker.py --dataset bail --teacher GraphSAGE --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:3
We have 18876 nodes.
We have 642616 edges.
Loading cascades...
 51%|██████████████████████████████████████████████████████████████████████▉                                                                    | 510/1000 [02:08<02:00,  4.06it/s]Stop!!!
 51%|██████████████████████████████████████████████████████████████████████▉                                                                    | 510/1000 [02:08<02:03,  3.97it/s]
Optimization Finished!
Total time elapsed: 87.0457s
Final results:
| Variant             | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|---------------------|---------------|---------------|---------------|---------------|
| ('bail', 'PLP', 42) | 0.8901±0.0000 | 0.8986±0.0000 | 0.0625±0.0000 | 0.0257±0.0000 |
```

### 7. An example with AKD+GCN under EO

Run

```
python train_dgl.py --dataset bail --teacher GCN --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 18876 nodes.
We have 642616 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:13<00:00, 76.24it/s]
Optimization Finished!
Total time elapsed: 8.5631s

| Variant             | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|---------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GCN', 42) | 0.2594±0.0000 | 0.9231±0.0000 | 0.0618±0.0000 | 0.0193±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset bail --teacher GCN --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 92.308%
Teacher Parity SCORE: 0.06178673506462984
Teacher Equality SCORE: 0.019254548666313376
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [02:20<00:00,  4.26it/s]

Final results:
| Variant            | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|--------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GCN', 1) | 0.8904±0.0000 | 0.8970±0.0000 | 0.0553±0.0000 | 0.0276±0.0000 |
```

### 8. An example with AKD+SAGE under EO

Run

```
python train_dgl.py --dataset bail --teacher GraphSAGE --framework GraphAKD
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
We have 18876 nodes.
We have 642616 edges.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:12<00:00, 81.56it/s]
Optimization Finished!
Total time elapsed: 8.2781s

| Variant                   | TestLoss      | TestAcc       | DeltaSP       | DeltaEO       |
|---------------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GraphSAGE', 42) | 0.3081±0.0000 | 0.9147±0.0000 | 0.0679±0.0000 | 0.0312±0.0000 |
```

After obtaining the teacher knowledge, then we enter the directory

```
cd ../GraphAKD/GraphAKD/GraphAKD-main/node-level/stu-gcn
```

and run

```
python train.py --dataset bail --teacher GraphSAGE --metric eo
```

Based on a fixed seed, we present the sample log as follows.

```
Device:  cuda:0
Teacher Test SCORE: 91.471%
Teacher Parity SCORE: 0.06787802435114115
Teacher Equality SCORE: 0.031160572337042947
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [02:27<00:00,  4.07it/s]

Final results:
| Variant                  | TestAcc       | ValAcc        | DeltaSP       | DeltaEO       |
|--------------------------|---------------|---------------|---------------|---------------|
| ('bail', 'GraphSAGE', 1) | 0.8965±0.0000 | 0.8986±0.0000 | 0.0565±0.0000 | 0.0317±0.0000 |
```



