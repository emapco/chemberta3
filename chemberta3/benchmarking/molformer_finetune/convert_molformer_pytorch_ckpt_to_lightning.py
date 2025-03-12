import re
import torch
import argparse
from transformers.utils import logging

logging.set_verbosity_info()

# Reverse the mapping rules
REVERSE_RULES = [
    (r"molformer\.embeddings\.word_embeddings", r"tok_emb"),
    (
        r"molformer\.encoder\.layer\.(\d+)\.attention\.self\.feature_map\.weight",
        r"blocks.layers.\1.attention.inner_attention.feature_map.omega",
    ),
    (
        r"molformer\.encoder\.layer\.(\d+)\.attention\.self\.(query|key|value)",
        r"blocks.layers.\1.attention.\2_projection",
    ),
    (r"molformer\.encoder\.layer\.(\d+)\.attention\.output\.dense",
     r"blocks.layers.\1.attention.out_projection"),
    (r"molformer\.encoder\.layer\.(\d+)\.attention\.output\.LayerNorm",
     r"blocks.layers.\1.norm1"),
    (r"molformer\.encoder\.layer\.(\d+)\.intermediate\.dense",
     r"blocks.layers.\1.linear1"),
    (r"molformer\.encoder\.layer\.(\d+)\.output\.dense",
     r"blocks.layers.\1.linear2"),
    (r"molformer\.encoder\.layer\.(\d+)\.output\.LayerNorm",
     r"blocks.layers.\1.norm2"),
    (r"molformer\.LayerNorm", r"blocks.norm"),
    (r"lm_head\.transform\.dense", r"lang_model.embed"),
    (r"lm_head\.transform\.LayerNorm", r"lang_model.ln_f"),
    (r"lm_head\.decoder", r"lang_model.head"),
]

for i, (find, replace) in enumerate(REVERSE_RULES):
    REVERSE_RULES[i] = (re.compile(find), replace)


def convert_pytorch_to_lightning_checkpoint(pytorch_checkpoint_path: str,
                                            lightning_checkpoint_path: str):
    # Load the PyTorch model checkpoint
    checkpoint = torch.load(pytorch_checkpoint_path,
                            map_location='cuda',
                            weights_only=True)

    # When using Distributed Data Parallel (DDP) for training models, PyTorch automatically
    # wraps model parameters in a module. prefix. This can cause issues when loading or
    # saving model states because the key names in state_dict differ from their original
    # single-GPU counterparts. To address this, model_state_dict is updated by removing
    # the "module." prefix when saving or loading models.

    checkpoint['model_state_dict'] = {
        key.replace("module.", ""): value
        for key, value in checkpoint['model_state_dict'].items()
    }

    new_state_dict = {}
    for key, val in checkpoint['model_state_dict'].items():
        for find, replace in REVERSE_RULES:
            if find.search(key) is not None:
                new_state_dict[find.sub(replace, key)] = val
                break

    # Save as a Lightning checkpoint
    lightning_checkpoint = {"state_dict": new_state_dict}
    print(f"Saving Lightning checkpoint to {lightning_checkpoint_path}")
    torch.save(lightning_checkpoint, lightning_checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_checkpoint_path",
                        required=True,
                        type=str,
                        help="Path to the PyTorch checkpoint.")
    parser.add_argument("--lightning_checkpoint_path",
                        required=True,
                        type=str,
                        help="Path to save the Lightning checkpoint.")
    args = parser.parse_args()
    convert_pytorch_to_lightning_checkpoint(args.pytorch_checkpoint_path,
                                            args.lightning_checkpoint_path)
