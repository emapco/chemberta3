import torch
import pandas as pd
import tempfile
from transformers import RobertaForMaskedLM, RobertaConfig
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from transformers.data.data_collator import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
import ray
from ray import train
import ray.train.torch
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.runtime_env import RuntimeEnv

def train_loop_per_worker(config):
    world_rank = train.get_context().get_world_rank()
    local_rank = train.get_context().get_local_rank()
    device = ray.train.torch.get_device()

    train_data_shard = train.get_dataset_shard("train")
    train_dataloader = train_data_shard.iter_batches(batch_size=32)

    tokenizer_path = 'seyonec/PubChem10M_SMILES_BPE_60k'
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    model = RobertaForMaskedLM(RobertaConfig(vocab_size=tokenizer.vocab_size))
    model = ray.train.torch.prepare_model(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(config["num_epochs"]):
        for i, batch in enumerate(train_dataloader):
            print (i)
            optimizer.zero_grad()
            inputs = batch['smiles']
            tokens = tokenizer(inputs.tolist(),
                               padding=True,
                               return_tensors="pt")
            tokens_device = tokens['input_ids'].device
            inputs, labels = data_collator.torch_mask_tokens(
                tokens['input_ids'])
            inputs = {
                'input_ids': inputs.to(device),
                'labels': labels.to(device),
                'attention_mask': tokens['attention_mask'].to(device),
            }
            outputs = model(**inputs)
            loss = outputs.get("loss")
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().item()

            metrics = {"loss": loss, "epoch": epoch, "iteration": i}

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                data = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss
                }
                torch.save(data, "ckpt.pt")
                ray.train.report(metrics,
                                 checkpoint=Checkpoint.from_directory(
                                     temp_checkpoint_dir))

if __name__ == '__main__':
    use_gpu = True
    num_workers = 4

    train_dataset = ray.data.read_csv('s3://chemberta3/datasets/zinc1m.csv')

    train_loop_config = {"num_epochs": 1}
    torch_config = TorchConfig(backend='nccl', timeout_s=600)
    ckpt_config = CheckpointConfig(num_to_keep=5,
                                   checkpoint_score_attribute='loss',
                                   checkpoint_score_order='min')
    run_config = RunConfig(
        checkpoint_config=ckpt_config,
        name='test',
        storage_path='s3://chemberta3/chemberta-test/zinc1m-mlm-nw-4-bs-32-s3-data/')

    trainer = TorchTrainer(train_loop_per_worker=train_loop_per_worker,
                           train_loop_config=train_loop_config,
                           datasets={"train": train_dataset},
                           scaling_config=ScalingConfig(
                               num_workers=num_workers, use_gpu=use_gpu,
                               resources_per_worker = {"GPU": 1}),
                           run_config=run_config,
                           torch_config=torch_config)
    result = trainer.fit()
    print('result path is ', result.path)
