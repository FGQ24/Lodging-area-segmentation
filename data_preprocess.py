# # # 别动啦！！！
from torch.utils.data import ConcatDataset
from torchvision import transforms
from Mydataset import MyDataset


def data_augment(root_dir, img_dir, label_dir):
    data0 = MyDataset(root_dir=root_dir,
                      img_dir=img_dir,
                      label_dir=label_dir,
                      transformimg=transforms.Compose([
                          # transforms.Grayscale(),
                          transforms.ToTensor(),
                          transforms.Resize([512, 512])]),
                      transformlab=transforms.Compose([
                          # transforms.Grayscale(),
                          transforms.ToTensor(),
                          transforms.Resize([512, 512])])
                      )
    # 数据增强
    # 随机水平翻转
    augmented_data1 = MyDataset(root_dir=root_dir,
                                img_dir=img_dir,
                                label_dir=label_dir,
                                transformimg=transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=1),
                                    # transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Resize([512, 512])]),
                                transformlab=transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=1),
                                    # transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Resize([512, 512])])
                                )
    # 随机垂直翻转
    augmented_data2 = MyDataset(root_dir=root_dir,
                                img_dir=img_dir,
                                label_dir=label_dir,
                                transformimg=transforms.Compose([
                                    transforms.RandomVerticalFlip(p=1),
                                    # transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Resize([512, 512])]),
                                transformlab=transforms.Compose([
                                    transforms.RandomVerticalFlip(p=1),
                                    # transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Resize([512, 512])])
                                )
    # 随机视角
    augmented_data3 = MyDataset(root_dir=root_dir,
                                img_dir=img_dir,
                                label_dir=label_dir,
                                transformimg=transforms.Compose([
                                    # transforms.Grayscale(),
                                    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                                    transforms.ToTensor(),
                                    transforms.Resize([512, 512])]),
                                transformlab=transforms.Compose([
                                    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                                    # transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Resize([512, 512])])
                                )
    # # 旋转 60 度
    # augmented_data4 = MyDataset(root_dir=root_dir,
    #                             img_dir=img_dir,
    #                             label_dir=label_dir,
    #                             transformimg=transforms.Compose([
    #                                 # transforms.Grayscale(),
    #                                 transforms.RandomRotation(degrees=60),
    #                                 transforms.Resize([512, 512]),
    #                                 transforms.ToTensor()]),
    #                             transformlab=transforms.Compose([
    #                                 transforms.RandomRotation(degrees=60),
    #                                 transforms.Resize([512, 512]),
    #                                 # transforms.Grayscale(),
    #                                 transforms.ToTensor()])
    #                             )

    if root_dir in ['data/train_data', 'data1/train_data', 'data2/train_data']:
        data = ConcatDataset(
            [data0, augmented_data1, augmented_data2, augmented_data3])
    elif root_dir in ['data/test_all', 'data/test_data/sim', 'data/test_data/dif',
                      'data1/test_all', 'data1/test_data/sim', 'data1/test_data/dif',
                      'data2/test_all', 'data2/test_data/sim', 'data2/test_data/dif',
                      'data2/bigall', 'data1/test_data/difmildblur', 'data1/test_data/simmildblur', 'cropped_images/3',
                      'cropped_images/5', 'cropped_images/6', 'cropped_images/7', 'cropped_images/9',
                      'cropped_images/12', 'cropped_images/14', 'cropped_images/18','data1/test_data/all',
                      'data1/test_data/diflight', 'data1/test_data/simlight', 'data1/test_data/difocclusion',
                      'data1/test_data/simocclusion']:
        data = data0
    else:
        raise ValueError('Invalid root_dir')

    return data
