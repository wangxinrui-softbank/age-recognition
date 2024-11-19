import os
import io
import cv2
import torch
import numpy as np
#import picamera

from iresnet import iresnet50
from model.model import RegressionResMLP
from matplotlib import pyplot as plt
from FaceInference import Detector, Face
from torchvision import transforms
#from Faceinference.mobile_lmks.landmarks import Face


def load_age_model():
    extractor = iresnet50(False, fp16=True, num_features=512)
    e_weight = torch.load('irse50_jcv.pt')
    extractor.load_state_dict(e_weight)
    extractor.eval()

    model = RegressionResMLP(dropout=0, num_residuals_per_block=2, 
                            num_blocks=4, num_classes=79, num_initial_features=512)
    m_weight = torch.load('best_mlp_6.93.pth', map_location='cpu')
    model.load_state_dict(m_weight)
    model.eval()

    return extractor, model


def load_face_model():
    from easydict import EasyDict as edict
    args = edict()
    args.trained_model = 'Faceinference/demo/retinaface.pth'
    args.cpu = True
    detector = Detector(args)

    kp_model_path = 'Faceinference/demo/kp_model.pth'
    alignment_models = torch.load(kp_model_path, weights_only=True)
    face = Face(alignment_models, is_cpu=args.cpu)

    return detector, face


def get_ref_landmarks():
    detector, face = load_face_model()
    image = cv2.imread('frontal.png')
    bboxes = detector.predict(image)
    
    x0, y0, x1, y1 = bboxes[0][:4]
    xc, yc, r = (x0+x1)/2, (y0+y1)/2, (x1-x0)/2
    yc = yc - 0.15*r
    newx0, newx1 = int(xc-1.05*r), int(xc+1.05*r)
    newy0, newy1 = int(yc-1.05*r), int(yc+1.05*r)
    
    
    results = face.predict(image, bboxes[0])
    landmarks = results['landmarks106']
    '''
    print(newx0, newx1, newy0, newy1, int(2.1*r))
    print(np.shape(landmarks))
    cv2.rectangle(image, (newx0, newy0), (newx1, newy1), (0, 0, 255), 2)
    for(x, y) in landmarks:
        #print(x, y)
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), 2)
    cv2.imwrite('output.jpg', image)
    '''
    landmarks[:, 0] -= newx0
    landmarks[:, 1] -= newy0
    landmarks = np.array(112 * landmarks/(2.1*r)).astype(np.float32)
    np.save('ref_landmarks.npy', landmarks)


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))
    R = np.dot(U, Vt).T
    s = s2/s1
    sR = s*R
    c1 = c1.reshape(2, 1)
    c2 = c2.reshape(2, 1)
    T = c2 - np.dot(sR, c1)
    trans_mat = np.hstack([sR, T])
    return trans_mat


def affine_with_kp(image1, h, w, kp1, kp2):
    trans_matrix = transformation_from_points(kp1, kp2)
    image1 = cv2.warpAffine(image1, trans_matrix, (h, w))
    return image1


def process_testset():
    ref_landmarks = np.load('ref_landmarks.npy')
    detector, face = load_face_model()
    folder = '/Users/wangx130/Downloads/dataset/age_dataset/images'
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            name = path.split('/')[-1]
            save_path = os.path.join('testset_crop', name)
            try:
                image = cv2.imread(path)
                bboxes = detector.predict(image)
                results = face.predict(image, bboxes[0])
                landmarks = results['landmarks106']
                output = affine_with_kp(image, 112, 112, landmarks, ref_landmarks)
                cv2.imwrite(save_path, output)
            except:
                pass


if __name__ == '__main__':
    
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    detector, face = load_face_model()
    extractor, model = load_age_model()
    ref_landmarks = np.load('ref_landmarks.npy')
    
    cap = cv2.VideoCapture(0)
    plt.ion()
    figure = plt.figure('frame')
    while True:
        _, frame = cap.read()
        
        try:
            bboxes = detector.predict(frame)
            results = face.predict(frame, bboxes[0])
            landmarks = results['landmarks106']
            crop = affine_with_kp(frame, 112, 112, landmarks, ref_landmarks)
            crop = transform(crop).unsqueeze(0)
            vector = extractor(crop)
            pred = model(vector).squeeze().detach().numpy()
            pred = np.argmax(pred)
            x0, y0, x1, y1 = bboxes[0][:4]
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
            frame = cv2.putText(frame, 'Age pred:{}'.format(pred+3), (int(x0), int(y0)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                
        except:
            pass
        
        ax1 = figure.add_subplot(1, 1, 1)
        ax1.axis('off')
        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.pause(0.001)

    plt.show()
    plt.ioff()
    
    






