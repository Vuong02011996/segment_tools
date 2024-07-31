# plt.show not show
+ Add: plt.switch_backend('WebAgg') # ['TKAgg','GTKAgg','Qt4Agg','WXAgg']

# Cannot load backend 'TKAgg' which requires the 'tk' interactive framework, as 'headless' is currently running

# Outof memory cuda
+ Add: torch.cuda.set_device(1)