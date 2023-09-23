## 简介
本文是以 MNIST 数据集来完成对图片任务分类的介绍  
MNIST 数据集的内容为手写数字，其在计算机视觉领域的地位不亚于 "hello world"  
数据如下所示
![image](https://github.com/CloseYYLab/YiliYili/assets/56760687/9cf61d3c-08f3-4be0-bdaf-05ffe034f789)

## 数据集下载及处理
使用 torchvision 可以下载 MNIST 数据集  
```python
root_dir = 'your_path' # 这里请转换成您的文件夹地址

train_data=torchvision.datasets.MNIST(
    root=root_dir,
    train=True,
    download=True
)
test_data=torchvision.datasets.MNIST(
    root=root_dir,
    train=False,
    download=True
)
```

得到的数据格式并非通用的数据格式，需要执行以下代码块就可得到本文所需要的数据集(该部分内容并不是本文重点，因此只需要运行即可)  

完整代码见下
```python
import os
from skimage import io
import torchvision
import torchvision.datasets.mnist as mnist

root_dir = 'your_path' # 这里请转换成您的文件夹地址

train_data=torchvision.datasets.MNIST(
    root=root_dir,
    train=True,
    download=True
)
test_data=torchvision.datasets.MNIST(
    root=root_dir,
    train=False,
    download=True
)

root_dir += 'MNIST/raw'
os.makedirs(root_dir,exist_ok = True)

train_set = (
    mnist.read_image_file(os.path.join(root_dir, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root_dir, 'train-labels-idx1-ubyte'))
        )
test_set = (
    mnist.read_image_file(os.path.join(root_dir, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root_dir, 't10k-labels-idx1-ubyte'))
        )
print("training set :",train_set[0].size())
print("test set :",test_set[0].size())

def convert_to_img(train=True):
    if(train):
        f=open(root_dir+'train.txt','w')
        data_path=root_dir+'/train/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
            img_path=data_path+str(i)+'.jpg'
            io.imsave(img_path,img.numpy())
            f.write(img_path+' '+str(label)+'\n')
        f.close()
    else:
        f = open(root_dir + 'test.txt', 'w')
        data_path = root_dir + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            img_path = data_path+ str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()

convert_to_img(True)#转换训练集
convert_to_img(False)#转换测试集
```
运行完毕后，会在指定的文件夹目录下生成一个MNIST文件夹，其子文件夹/raw中包含/test和/train两个图片文件夹，以及rawtest.txt和rawtrain.txt两个标签文件  
至此，数据准备工作就告一段落

## 创建dataset
dataset 的作用是对整个数据集进行整理，排序。如果说原始数据集是分散在各个同学手里的作业，那么 dataset 就是收作业的课代表。课代表按照学号来把所有的作业本收集起来，然后提供给老师。这也就是 dataset 的作用  
dataset 的构建需要继承 torch 的 Dataset 类并进行重写  

重写的主要有以下三个方法：`__init__`, `__getitem__` 和 `__len__`   

- `__init__` : 负责传递图片和标签所对应的文件夹路径
- `__getitem__` : 负责返回一张图片和其对应的标签
- `__len__`  : 负责返回数据集的长度


图片分类任务是最基本的计算机视觉任务，相较于目标检测及其他任务来说，他的标签较为简单。常用的形式有两种：  
1. 以类别 id 作为文件夹的名字，每个文件夹下包含着同一种类别的图片
2. 使用一个 txt 文本来保存全部的标签，文件内的每一行是 图片路径：标签

本文所使用的方法是第二种方法，也就是有训练集标签.txt和测试集标签.txt两个文件  
在处理第一种方法的数据集时，可以在 `__getitem__` 方法内对图片的路径进行处理来得到标签，也可以生成 txt 文件来获得全部的标签，各位同学按照自己的喜好来选择即可  


