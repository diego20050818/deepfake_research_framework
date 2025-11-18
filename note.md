# 实验日志
## 2025年11月11日
初步查看了浩恩的框架，，探讨了可行性之后决定大干一场
目前是处于处理数据集的 阶段
构想中的数据格式应该是

```
    dataset:
        train
        test
        val

        train.csv
        test.csv
        val.csv
```
其中每一个csv文件
```
   train.csv:

   img_name,label
```   
但是关键是，人脸检测这一块谁给我啊a


添加关于AUC损失的支持
