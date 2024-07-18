from src.vit import ViT, ViTConfig

import torch
import time
import math
import inspect


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# grad
# total_batch_size = 256
# batch_size = 32

# assert total_batch_size % batch_size == 0, 'make sure that total_batch_size is divisible by batch_size'
# grad_accum_steps = total_batch_size / batch_size


# for step in range(100):
#     begin_time = time.time()
#     optimizer.zero_grad()
#     loss_accum = 0.0
#     # grad accum
#     for micro_step in range(grad_accum_steps):
#         test_tensor = torch.rand((batch_size, 3, 224, 224)).to(device)         # emulate batch of batch_size (3 channel) images
#         with torch.autocast(device_type=device, dtype=torch.bfloat16):         # to enable BF16 precision  ### 2
#             out = model(test_tensor)
#             loss = None
#         
#         loss = loss / grad_accum_steps
#         loss_accum += loss.detach()
#         loss.backward()

#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#     lr = get_lr(step)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
    
#     optimizer.step()
#     torch.cuda.synchronize()
#     print("Finished in ", time.time() - begin_time)




# optimizer = torch.optim.AdamW(lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

# max_lr = 3e-4
# min_lr = max_lr * 0.1
# warmup_steps = 10
# max_steps = 50

# def configure_optimizer(weight_decay, learning_rate, device, model):
#     param_dict = {pn: p for pn, p in model.named_parameters()}
#     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

#     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
#     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
#     optim_groups = [
#         {'params':decay_params, 'weight_decay':weight_decay},
#         {'params':nodecay_params, 'weight_decay':0.0}
#     ]

#     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
#     use_fused = fused_available and 'cuda' in device
#     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
#     return optimizer


### lr




