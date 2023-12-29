from torch.utils import data
import natsort
import os
from PIL import Image
from torchvision import transforms as T


def get_loader(data_path, im_size, batch_size, num_workers, kind='train', shu_flag=1, im_flag='bmp'):
    dataset = GetDataset(data_path=data_path, im_size=im_size, kind=kind, im_format=im_flag)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=shu_flag,)
    return data_loader


def get_data_path(data_path, im_format):
    t = []      #分别存储train、test、valid中的每个文件相对路径名，
    data_path_data = os.path.join(data_path, 'data')
    for path, dir, filelist in os.walk(data_path_data):
        for filename in filelist:
            if filename.endswith(str(im_format)):
                temp = list(map(str, path))
                if temp[-1] == '/':
                    temp = temp[:-1]
                temp = ''.join(temp)
                t.append(temp + '/' + filename)
    t1 = []
    data_path_label = os.path.join(data_path, 'label')
    for path, dir, filelist in os.walk(data_path_label):
        for filename in filelist:
            if filename.endswith(str(im_format)):
                temp = list(map(str, path))
                if temp[-1] == '/':
                    temp = temp[:-1]
                temp = ''.join(temp)
                t1.append(temp + '/' + filename)
    im = natsort.natsorted(list(set(t).difference(set(t1))))
    gt = natsort.natsorted(t1)
    return im, gt


def image_trans(im, kind, im_size, norm=False):
    aug = T.Compose([T.Resize(im_size),
                     T.ToTensor(),T.Grayscale()])
    ret = aug(im)
    if norm:
        n = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ret = n(ret)
    return ret

class GetDataset(data.Dataset):
    def __init__(self, data_path, im_size, kind, im_format):
        paths = get_data_path(data_path, im_format)
        self.im_paths = paths[0]
        self.gt_paths = paths[1]
        self.image_size = im_size
        self.kind = kind
        print("Images in {} path :{}".format(self.kind, len(self.im_paths)))

    def __getitem__(self, index):
        im_path = self.im_paths[index]
        gt_path = self.gt_paths[index]

        im = image_trans(Image.open(im_path), self.kind, self.image_size)
        gt = image_trans(Image.open(gt_path), self.kind, self.image_size)
        return im, gt

    def __len__(self):
        return len(self.im_paths)


if __name__ == '__main__':
    im = Image.open('85.png')
    im_t = image_trans(im, 512)
    plt.subplot(221)
    plt.imshow(im, cmap='gray')
    plt.subplot(222)
    plt.imshow(np.array(im_t).squeeze(0), cmap='gray')
    plt.show()










































