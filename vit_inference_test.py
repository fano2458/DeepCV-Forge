from src.vit import ViT, ViTConfig

import torch
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_float32_matmul_precision('high')  # to enable TF32 precision         ### 1


model = ViT(ViTConfig()).from_pretrained('facebook/deit-tiny-patch16-224').eval().to(device)
model = torch.compile(model)                        # jit-compiling  ### 3     # not supported on python 3.12+

with torch.no_grad():
    for i in range(50):
        test_tensor = torch.rand((1024, 3, 224, 224)).to(device)               # emulate batch of 1024 (3 channel) images
        begin_time = time.time()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):         # to enable BF16 precision  ### 2
            out = model(test_tensor)
        torch.cuda.synchronize()
        print("Finished in ", time.time() - begin_time)
