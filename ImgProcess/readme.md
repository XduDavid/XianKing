代码参考

https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec
原作者完整体代码

https://github.com/tomahim/py-image-dataset-generator
skimage库调用

https://scikit-image.org/docs/stable/auto_examples/index.html

操作步骤:
将照片放在`./0`下执行clean.py剔除小于100*100的照片
再将过滤后的照片移至`./2`下执行即可，需要其他的效果参考skimage的例程
**建议保存为png，保存为jpg可能会因为存在alpha通道信息而保存失败**