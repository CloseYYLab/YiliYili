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
