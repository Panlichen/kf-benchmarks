
import tensorflow as tf
import tensorflow_datasets as tfds

# 设置要加载的样本数量
num_samples = 1000  # 你可以根据需要调整这个数字

# 加载 ImageNet 子集
dataset, info = tfds.load(
    'imagenet2012_subset/1pct',  # 使用 10% 的子集
    split='train[:{}]'.format(num_samples),  # 只获取指定数量的样本
    with_info=True,
    as_supervised=True,
)

# 打印数据集信息
print(info)

# 查看几个样本
for image, label in dataset.take(5):
    print("Image shape:", image.shape)
    print("Label:", label.numpy())
    