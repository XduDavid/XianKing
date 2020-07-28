# XianKing

1、将训练生成的`XianKing_4w4a.pt`放到此文件夹中。

2、运行`torch_export.py`文件，会在当前目录生成`XianKing_4w4a.npz`和`config.json`文件。

3、运行`XianKing_param_gen.py`文件，设定各层输入输出并行度，运行后会在`./param/hls/`文件夹中生成`config.h`和`param.h`，供HLS模块调用。

