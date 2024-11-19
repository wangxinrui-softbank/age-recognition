import os
import cv2
import ast
import json
import torch
import random
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms


def align(image, landmarks, scale=2.5):
    lmks5 = [landmarks[i] for i in [74, 77, 45, 84, 90]]
    left_eye = np.array([lmks5[0]['x'], lmks5[0]['y']], dtype=np.float32)
    right_eye = np.array([lmks5[1]['x'], lmks5[1]['y']], dtype=np.float32)
    nose = np.array([lmks5[2]['x'], lmks5[2]['y']], dtype=np.float32)
    eye_width = right_eye - left_eye
    angle = np.arctan2(eye_width[1], eye_width[0])
    center = nose
    alpha = np.cos(angle)
    beta = np.sin(angle)
    w = np.sqrt(np.sum(eye_width**2)) * scale
    m = [[alpha, beta, -alpha * center[0] - beta * center[1] + w * 0.5],
        [-beta, alpha, beta * center[0] - alpha * center[1] + w * 0.5]]
    aligned = cv2.warpAffine(image, np.array(m), [int(w), int(w)])
    return aligned


def align106(image, landmarks, scale=2.5):
    lmks5 = [landmarks[i] for i in [74, 77, 45, 84, 90]]
    left_eye = np.array([lmks5[0][0], lmks5[0][1]], dtype=np.float32)
    right_eye = np.array([lmks5[1][0], lmks5[1][1]], dtype=np.float32)
    nose = np.array([lmks5[2][0], lmks5[2][1]], dtype=np.float32)
    eye_width = right_eye - left_eye
    angle = np.arctan2(eye_width[1], eye_width[0])
    center = nose
    alpha = np.cos(angle)
    beta = np.sin(angle)
    w = np.sqrt(np.sum(eye_width**2)) * scale
    m = [[alpha, beta, -alpha * center[0] - beta * center[1] + w * 0.5],
        [-beta, alpha, beta * center[0] - alpha * center[1] + w * 0.5]]
    aligned = cv2.warpAffine(image, np.array(m), [int(w), int(w)])
    return aligned



def age_map(age):
    if age <= 3:
        return 0
    elif age > 3 and age <= 80:
        return int(age + 0.5) - 3
    else:
        return 78



class TXTDataset(torch.utils.data.Dataset):
    def __init__(self, txt_dir, clf=False, size=112):
        self.age_map = {i: i for i in range(101)}
        #self.df = pd.read_csv(csv_dir)
        self.clf = clf
        self.file = open(txt_dir, 'r')
        self.file_list = self.file.read().split("\n")[:-2]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
 

    def __len__(self) -> int:
        return len(self.file_list)


    def __getitem__(self, idx: int) -> tuple:
        row = self.file_list[idx]
        image_path, fr_info_path = row.split(' ')
        image = cv2.imread(image_path)[:, :, ::-1]
        info = pd.read_pickle(fr_info_path)
        landmarks = json.loads(info['landmark'])
        #json_string = open(info['attributes'], "r")
        json_string = ast.literal_eval(info['attributes'])
        attr = json.loads(json_string)
        age = attr['st_age_value'][0]['confidence']

        if not self.clf:
            y = np.float32([age])
        else:
            y = int(age + 0.5) - 2
        image = align(image, landmarks)
        x = self.transform(image)
        return x, y


class VALDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, size=112):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
 

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
        name = self.file_list[idx]
        image_path = os.path.join(self.data_dir, name)
        image = cv2.imread(image_path)[:, :, ::-1]
        age = float(name.split('_')[2])
        #y = int(age + 0.5) - 2
        y = age_map(age)
        x = self.transform(image)
        return x, y


def TXTDataLoader(txt_dir, batch_size, shuffle, num_workers, clf=False):
    dataset = TXTDataset(txt_dir=txt_dir, clf=clf)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            num_workers=num_workers, shuffle=shuffle)
    return dataloader


def VALDataLoader(data_dir, batch_size, shuffle, num_workers):
    dataset = VALDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            num_workers=num_workers, shuffle=shuffle)
    return dataloader


def process_val():
    from face_sdk.face_sdk import get_keypoints
    folder = '/home/ec2-user/dataset/valset_age'
    count0, count1 = 0, 0
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            count0 += 1
            path = os.path.join(root, name)
            image = cv2.imread(path)
            try:
                landmarks = get_keypoints(image)[:106]
                image = align106(image, landmarks)
                image = cv2.resize(image, (112, 112))
                #save_path = path.replace('/valset_age/', '/valset_age_crop/')
                save_path = os.path.join('/home/ec2-user/dataset/valset_age_crop', name)
                cv2.imwrite(save_path, image)
                count1 += 1
            except:
                print(name)
                
    print(count0, count1)




if __name__ == '__main__':
    '''
    from tqdm import tqdm
    path = "/share5/dataset/webface42m/webface42m_val1m.txt"
    val_loader = TXTDataLoader(path, 128, False, 8)
    for batch in tqdm(val_loader):
        pass
    '''

    process_val()
