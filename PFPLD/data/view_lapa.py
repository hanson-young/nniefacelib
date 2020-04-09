import os
import cv2
import glob
import insightface
import imutils
import numpy as  np
import tqdm

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[7])
    B = np.linalg.norm(eye[2] - eye[6])
    C = np.linalg.norm(eye[3] - eye[5])
    D = np.linalg.norm(eye[0] - eye[4])
    ear = (A + B + C) / (3.0 * D)
    return ear

set_part = 'train'
print("set_part : ", set_part)
root_path = '/media/data4T1/hanson/Landmarks/LaPa'
label_txt = os.path.join(root_path, 'lapa_' + set_part + '_label.txt')
drap_index = [56, 57, 58, 64, 65, 66, 75, 84]
imgslist = glob.glob(os.path.join(root_path, set_part + '/images', '*.jpg'))
retina = insightface.model_zoo.get_model('retinaface_mnet025_v1')
retina.prepare(-1, 0.4)
with open(label_txt, 'w') as wf:
    for img_path in tqdm.tqdm(imgslist):
        landm_path = os.path.join(root_path, set_part + '/landmarks',img_path.split('/')[-1].replace('.jpg', '.txt'))

        with open(landm_path, 'r') as rf:
            landms = rf.readlines()
        img = cv2.imread(img_path)
        det_img = imutils.resize(img, width=480)
        r = float(img.shape[1]) / 480.
        lanmark = []

        for i in range(1, len(landms)):
            if i in drap_index:
                continue
            x, y = landms[i].strip().split()
            lanmark.append(float(x))
            lanmark.append(float(y))
            cv2.circle(img, (int(float(x)), int(float(y))),1,(255, 0, 0), 1)


        leye = []
        reye = []
        leye.append(np.array([lanmark[60 * 2 + 0], lanmark[60 * 2 + 1]]))
        leye.append(np.array([lanmark[61 * 2 + 0], lanmark[61 * 2 + 1]]))
        leye.append(np.array([lanmark[62 * 2 + 0], lanmark[62 * 2 + 1]]))
        leye.append(np.array([lanmark[63 * 2 + 0], lanmark[63 * 2 + 1]]))
        leye.append(np.array([lanmark[64 * 2 + 0], lanmark[64 * 2 + 1]]))
        leye.append(np.array([lanmark[65 * 2 + 0], lanmark[65 * 2 + 1]]))
        leye.append(np.array([lanmark[66 * 2 + 0], lanmark[66 * 2 + 1]]))
        leye.append(np.array([lanmark[67 * 2 + 0], lanmark[67 * 2 + 1]]))

        reye.append(np.array([lanmark[68 * 2 + 0], lanmark[68 * 2 + 1]]))
        reye.append(np.array([lanmark[69 * 2 + 0], lanmark[69 * 2 + 1]]))
        reye.append(np.array([lanmark[70 * 2 + 0], lanmark[70 * 2 + 1]]))
        reye.append(np.array([lanmark[71 * 2 + 0], lanmark[71 * 2 + 1]]))
        reye.append(np.array([lanmark[72 * 2 + 0], lanmark[72 * 2 + 1]]))
        reye.append(np.array([lanmark[73 * 2 + 0], lanmark[73 * 2 + 1]]))
        reye.append(np.array([lanmark[74 * 2 + 0], lanmark[74 * 2 + 1]]))
        reye.append(np.array([lanmark[75 * 2 + 0], lanmark[75 * 2 + 1]]))

        leye = np.array(leye)
        l_ear = eye_aspect_ratio(leye)
        reye = np.array(reye)
        r_ear = eye_aspect_ratio(reye)
        for k in range(leye.shape[0]):
            x, y = leye[k]
            cv2.circle(img, (int(float(x)), int(float(y))),1,(0, 255, 0), 2)
            x, y = reye[k]
            cv2.circle(img, (int(float(x)), int(float(y))),1,(0, 255, 255), 2)

        if l_ear > 0.1 and r_ear > 0.1:
            continue
        bboxes, landmarks = retina.detect(det_img, threshold=0.5, scale=1.0)
        if bboxes.shape[0] >= 1:
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            img_center = det_img.shape[0] // 2, det_img.shape[1] // 2
            offsets = np.vstack(
                [(bboxes[:, 0] + bboxes[:, 2]) / 2 - img_center[1], (bboxes[:, 1] + bboxes[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:1]
            bboxes = bboxes[bindex, :]
            landmarks = landmarks[bindex, :]
        else:
            continue
        box = [int(bboxes[0][0] * r), int(bboxes[0][1] * r),int(bboxes[0][2] * r), int(bboxes[0][3] * r)]
        cv2.rectangle(img, (box[0], box[1]),(box[2], box[3]),(0,255,0),1)
        landmark_str = ' '.join(list(map(str, lanmark)))
        attributes_str = '0 0 0 0 0 0'
        path_str = 'LAPA_' + set_part + '/' + img_path.split('/')[-1]
        box_str = ' '.join(list(map(str, box)))
        label = '{} {} {} {}\n'.format(landmark_str, box_str, attributes_str, path_str)
        # if l_ear < 0.2 or r_ear < 0.2:
        wf.write(label)
        # print(label)
        print("left eye : " , l_ear)
        print("reft eye : ", r_ear)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)