# plt.show not show
+ Add: plt.switch_backend('WebAgg') # ['TKAgg','GTKAgg','Qt4Agg','WXAgg']

# Cannot load backend 'TKAgg' which requires the 'tk' interactive framework, as 'headless' is currently running

# Outof memory cuda
+ Add: torch.cuda.set_device(1)

# RuntimeError: No available kernel. Aborting execution.
+ https://github.com/facebookresearch/segment-anything-2/issues/48
+ OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = True, True, True
