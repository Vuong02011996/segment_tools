# plt.show not show
+ Add: plt.switch_backend('WebAgg') # ['TKAgg','GTKAgg','Qt4Agg','WXAgg']

# Cannot load backend 'TKAgg' which requires the 'tk' interactive framework, as 'headless' is currently running

# Outof memory cuda
+ Add: torch.cuda.set_device(1)

# RuntimeError: No available kernel. Aborting execution.
+ https://github.com/facebookresearch/segment-anything-2/issues/48
+ OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = True, True, True


# Error import label to cvat: Invalid RLE mask representation cvat
+ https://github.com/cvat-ai/cvat/issues/6487
+ Thanks a lot, both w and h should +1. And, I don't really know the CVAT version since I am not the person who labeled the image.

# error: no such file or directory: 'pycocotools/_mask.c' when running the makefile #661
+ mask.pyi add: def decompress(rleObjs: _EncodedRLE) -> _NDArrayFloat64: ...
+ mask.py add:
    def decompress(rleObjs):
    if type(rleObjs) == list:
        return _mask.decompress(rleObjs)
    else:
        return _mask.decompress([rleObjs])[0]

+ pip install cython did the trick for me