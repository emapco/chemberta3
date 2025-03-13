# Finetuning of MoLFormer model pre-trained using DeepChem on MoLFormer (IBM) finetuning pipeline.

Step 1: Convert MoLFormer model's DeepChem checkpoint to PyTorch Lightning format

The "convert_molformer_pytorch_ckpt_to_lightning.py" script can be used to convert the MoLFormer model's Deepchem format checkpoint to Pytorch Lightning format. 
It takes two arguments -
1. pytorch_checkpoint_path - Deepchem format pre-trained model checkpoint path
2. lightning_checkpoint_path - path to store the converted checkpoint (Lightning format)

A test example is provided at - 'benchmarking/tests/test_convert_molformer_pytorch_ckpt_to_lightning.py

Step 2: Setup MoLFormer repo 

Link to the repo - "https://github.com/IBM/molformer"
    
Step 3: Run finetuning scripts

Modify the seed_path and checkpoints_folder arguments in the finetuning scripts which can be found in the
"molformer/finetune" folder in molformer repo. 
Here seed_path is the path to the pre-trained checkpoint in the pytorch Lightning format, and checkpoints_folder stores the finetuned model checkpoint and scores for various epochs. 