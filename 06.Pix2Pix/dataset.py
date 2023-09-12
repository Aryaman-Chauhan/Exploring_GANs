from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config

class MultiDataset(Dataset):
    def __init__(self, root_dir, split=False, targetOnLeft=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.targetOnLeft = targetOnLeft
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files) if self.split==False else len(os.listdir(os.path.join(self.root_dir, self.list_files[0])))
    
    def __getitem__(self, index):
        if self.split==False:
            img_file = self.list_files[index]
            img_path = os.path.join(self.root_dir, img_file)
            image = np.array(Image.open(img_path))
            if self.targetOnLeft:
                input_image = image[:, 256:, :]
                target_image = image[:, :256, :]
            else:
                input_image = image[:, :256, :]
                target_image = image[:, 256:, :]

            return input_image, target_image
        else:
            output_img_file = os.listdir(os.path.join(self.root_dir,self.list_files[0]))[index]
            input_img_file = os.listdir(os.path.join(self.root_dir,self.list_files[1]))[index]
            output_img_path = os.path.join(self.root_dir, self.list_files[0], output_img_file)
            input_img_path = os.path.join(self.root_dir, self.list_files[1], input_img_file)
            # print(output_img_path, input_img_path)
            target_image = np.array(Image.open(output_img_path))
            input_image = np.array(Image.open(input_img_path))
            return input_image, target_image

    
    def aug_Image(self, input_image, target_image, extraAug=False):
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]
        if extraAug:
            input_image = config.transform_only_input(input_image)
            target_image = config.transform_only_mask(target_image)
        return input_image, target_image

def test():
    dataset1 = MultiDataset('facetocomics', split=True, targetOnLeft=True)
    print(len(dataset1))
    print(dataset1.__getitem__(1)[0].shape)
    dataset2 = MultiDataset('facades/train')
    print(len(dataset2))
    print(dataset2.__getitem__(1)[0].shape)

if __name__ == "__main__":
    test()