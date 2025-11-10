import torch, importlib
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("datastates import OK:", importlib.import_module("datastates"))
print("ckpt ext import OK:", importlib.import_module("datastates.ckpt.src"))