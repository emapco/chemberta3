Torch distributed

### ddp_cpu.py

Does not use an iterable dataset - IO intensive.

### ddp_cpu_iterable_dataset.y

Uses an iterable style dataset created via existing deepchem infra.

### ddp_iterable_dataset_single_gpu.py

Uses an iterable dataset taking into account the distributed processes.

## Multi-node training

Steps:
- Setup a cluster using the terraform scripts in `chemberta3/distributed/infra/multi-node`
- Split the data into the machines
- Use the `chemberta3/distributed/td/dc_multinode.py` test script for testing training
- On node 0, the script is invoked as `torchrun --nproc-per-node=1 --nnodes=2 --node-rank=0 --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=172.16.130.32:16000 dc_multinode.py` and on node 1, as `torchrun --nproc-per-node=1 --nnodes=2 --node-rank=1 --rdzv-id=456 --rdzv-backend=c10d --rdzv-endpoint=172.16.130.32:16000 dc_multinode.py`. Here, `172.16.130.32` is the private ipv4 address of node 0.

### Debugging tips
- Check whether DNS hostname and DNS resolution are enabled on the AWS VPC.
- Check whether the nodes can communicate using `ping`
