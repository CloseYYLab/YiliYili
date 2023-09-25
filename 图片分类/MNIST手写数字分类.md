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
            f.write(img_path+' '+str(int(label))+'\n')
        f.close()
    else:
        f = open(root_dir + 'test.txt', 'w')
        data_path = root_dir + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            img_path = data_path+ str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(int(label)) + '\n')
        f.close()

convert_to_img(True)#转换训练集
convert_to_img(False)#转换测试集
```
运行完毕后，会在指定的文件夹目录下生成一个MNIST文件夹，其子文件夹/raw中包含/test和/train两个图片文件夹，以及rawtest.txt和rawtrain.txt两个标签文件  
**至此，数据准备工作就告一段落**

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

`__init__`:加载全部的图片路径，并以字典的形式存储图片路径和对应的标签
```python
    def __init__(self,label_dir,image_dir):
        self.label_dir = label_dir
        self.image_dir = image_dir
        # 获取全部的图片并保存在list中
        self.image_path_list = os.listdir(image_dir)
        # 构建 image_name 和 label 的键值对
        self.label_dict = {}
        with open(self.label_dir,'r') as fr:
            datas = fr.readlines()
            for data in datas:
                tmp = data.split(' ')
                tmp_image_abs_path = tmp[0]
                tmp_label = int(tmp[1]) # 这里加 int 是为了去掉结尾换行符 '\n'
                self.label_dict[tmp_image_abs_path] = tmp_label
```

`__getitem__`:返回cv格式的图片和int类型的标签
```python
    def __getitem__(self, index):
        image_name = self.image_path_list[index]
        image_abs_path = os.path.join(self.image_dir,image_name)

        label = self.label_dict[image_abs_path]
        image = cv2.imread(image_abs_path)
        return image,label
```

`__len__`:返回数据集大小
```python
    def __len__(self):
        return len(self.image_path_list)
```

完整代码如下(附上运行代码)
```python
import torchvision
from torch.utils.data import Dataset
import cv2
import os

class learn_dataset(Dataset):
    def __init__(self,label_dir,image_dir):
        self.label_dir = label_dir
        self.image_dir = image_dir
        # 获取全部的图片并保存在list中
        self.image_path_list = os.listdir(image_dir)
        # 构建 image_name 和 label 的键值对
        self.label_dict = {}
        with open(self.label_dir,'r') as fr:
            datas = fr.readlines()
            for data in datas:
                tmp = data.split(' ')
                tmp_image_abs_path = tmp[0]
                tmp_label = int(tmp[1]) # 这里加 int 是为了去掉结尾换行符 '\n'
                self.label_dict[tmp_image_abs_path] = tmp_label
            
    def __getitem__(self, index):
        image_name = self.image_path_list[index]
        image_abs_path = os.path.join(self.image_dir,image_name)

        label = self.label_dict[image_abs_path]
        image = cv2.imread(image_abs_path)
        return image,label

    def __len__(self):
        return len(self.image_path_list)

if __name__ == '__main__':
    train_dataset = learn_dataset(image_dir='/home/jvm/yolov5/learn/MNIST/raw/train',
                                  label_dir='/home/jvm/yolov5/learn/MNIST/rawtrain.txt')
    test_dataset = learn_dataset(image_dir='/home/jvm/yolov5/learn/MNIST/raw/test',
                                  label_dir='/home/jvm/yolov5/learn/MNIST/rawtest.txt')
    image,label = train_dataset[0]
    
    print(label)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
```
运行结束后可以得到一个显示图片的窗口，在解释器的终端里会输出对应的标签(这里由于是服务器,不方便显示图片,就不再粘贴图片了)

### dataset 的炫酷版本
由于 dataset 的 `__init__` 函数会在创建时构建图片路径和标签的字典，这时候程序在内部运行，没有显示信息。如果我们想了解现在加载到哪一步了，应该怎么做呢？  
没错，我们可以使用进度条来提示现在加载到那一部分了。而进度条的使用也非常简单，只需要引入 `tqdm` 库就可以了  
首先，先执行安装语句
```python
pip install tqdm
```

tqdm 的使用非常简单，可以理解为对 for 循环又包了一层，例子见下
```python
from tqdm import tqdm
import time 

for i in tqdm(range(20), desc='tqdm test'):
    time.sleep(0.1)
```

其中 tqdm 语句的一个参数就是要运行的目标，通常是一个**可迭代对象**，而 desc 则是进度条前面的描述语句，一般来讲填这两个就可以了

tqdm()主要参数默认值与解释
- `iterable=None`:可迭代对象。如上一节中的range(20)
- `desc=None`:传入str类型，作为进度条标题。如上一节中的desc='It\'s a test'
- `total=None`:预期的迭代次数。一般不填，默认为iterable的长度。
- `leave=True`:迭代结束时，是否保留最终的进度条。默认保留。
- `file=None`:输出指向位置，默认是终端，一般不需要设置。
- `ncols=None`:可以自定义进度条的总长度
- `unit`:描述处理项目的文字，默认’it’，即100it/s；处理照片设置为’img’，则为100img/s
- `postfix`:以字典形式传入详细信息，将显示在进度条中。例如postfix={'value': 520}
- `unit_scale`:自动根据国际标准进行项目处理速度单位的换算，例如100000it/s换算为100kit/s

## DataLoader
dataset 是加载全部数据集，但是我们希望以分批的形式把他们送给模型，而不是每次都把全部数据集丢给网络   
就像老师本人并不亲自收作业，而是让几个课代表去收，收完了再送给老师  

dataset可以看作是学生提交的各个作业。而dataloader则相当于老师让课代表去收取作业，那么网络训练的整体过程可以理解为将所有作业按照一定的规则（例如按照学生姓名或学号排序）分成若干个小组，每次从一个小组中取出一定数量的作业进行批量处理和评分，然后再从下一个小组中取出一批作业，重复这个过程直到所有作业都被处理完毕。

dataloader 的主要参数有以下几个`dataset`,`batch_size`,`shuffle`,`num_workers`,`drop_last`

- `dataset`:指向的数据集
- `batch_size`:一个批次的大小，也就是加载几个数据
- `shuffle`:是否以乱序从数据集中加载数据
- `num_workers`:用来加载数据的线程数
- `drop_last`:如果剩下的数据不满足一个批次，是否丢掉

更多参数说明请参考 https://pytorch.org/docs/stable/data.html  

好的，那么现在为我们的训练集和测试集都创建一个 dataloader  
```python
from torch.utils.data import DataLoader
# 创建 dataloader
train_dataloader = DataLoader(dataset = train_dataset,batch_size = 10,shuffle=True,num_workers=2,drop_last=False)
test_dataloader = DataLoader(dataset = test_dataset,batch_size = 10,shuffle=True,num_workers=2,drop_last=False)
    
# 遍历 dataloader
for image,label in train_dataloader:
    print(image,label)
```
