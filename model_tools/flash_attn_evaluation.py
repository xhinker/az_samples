#%%
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

#%%
import torch, flash_attn
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
# print(flash_attn.__version__)
print("flash_attn cuda archs:", flash_attn._flash_attn_cuda.__dict__.get("__arch__", "unknown"))

#%%
import torch, flash_attn

#%%
import torch, flash_attn
from flash_attn.ops.fused_dense import fused_dense

B, H, L, D = 2, 8, 128, 128
q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attn.flash_attn_func(q, k, v)
print(out.shape)                 # (2, 8, 128, 128)  and runs <50 ms


#%%
import traceback, importlib
try:
    import flash_attn._flash_attn_cuda as _fa
    print("✅ extension loaded:", _fa.__file__)
except Exception as e:
    traceback.print_exc()          # shows the hidden ImportError / OSError
    
#%%
from flash_attn.ops.fused_dense import fused_dense
print(fused_dense)      # prints a <built-in function> object, not a stub