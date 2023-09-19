import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import cv2

from model.nets.mpunet import MPUNet
import torch.nn.functional as F
from post_proc import boundary_proc, xcx_proc,classical_boundary_proc
from post_water import water_post


def read_image(image_path):
    if Path(image_path).exists():
        image = Image.open(image_path).convert('L')
        image = np.array(image)
        return image
    else:
        return None


def inference(weight_path, root_path, save_path, use_post_process = False):
    """
    推理
    :param weight_path:  weight path
    :param root_path:  The root directory of the file you want to infer
    :param save_path:  Inference result storage directory
    :param use_post_process:
    :return:
    """
    root_path = Path(root_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    transform_feature = transforms.Compose([
        transforms.ToTensor()
    ])

    print('start inference')
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    print('use', device)

    net = MPUNet().to(device)
    if os.path.exists(weight_path):
        a, b = os.path.splitext(weight_path)
        if b == '.pth': 
            net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight_path).items()})
        else:
            print("Unsupported file type!")
        print('Model loaded successfully!')
    else:
        raise RuntimeError('Model not Found in weight_path!')

    net.eval()

    for image_path in root_path.iterdir():
        img = read_image(image_path)
        image = transform_feature(img)
        image = image.to(device)
        image = image.unsqueeze(dim=0)

        # label 
        img_name = image_path.name
        print(img_name)
        print(Path(root_path.parent,"test_boundary_label",img_name))
        label_xcx = cv2.imread(str(Path(root_path.parent,"test_xcx_label",img_name)),0)
        label_boundary = cv2.imread(str(Path(root_path.parent,"test_boundary_label",img_name)),0)


        out_tensor, xcx_tensor = net(image)
        out_tensor = F.softmax(out_tensor, dim = 1)
        xcx_tensor = F.softmax(xcx_tensor, dim = 1)

        out_tensor = out_tensor[0, :, :, :].cpu().detach().numpy().transpose(1,2,0)
        xcx_tensor = xcx_tensor[0, :, :, :].cpu().detach().numpy().transpose(1,2,0)

        out_img = np.argmax(out_tensor, axis=2).astype(np.uint8)
        xcx_img = np.argmax(xcx_tensor, axis=2).astype(np.uint8)
        if use_post_process: # post-process include prun and adaptative strategy
            classical_out_img = classical_boundary_proc(out_img) # classical_boundary_proc
            out_img = boundary_proc(out_img)
            xcx_img = xcx_proc(xcx_img)
            out_img[out_img>0] = 255
            xcx_img[xcx_img>0] = 255
            water_out_img = water_post(img, out_img, xcx_img, thresh_iou=0.9)
            water_out_img = boundary_proc(water_out_img)

        out_img[out_img>0] = 255
        xcx_img[xcx_img>0] = 255
        water_out_img[water_out_img>0] = 255

        
        label_name = image_path.name
        boundary_path = Path(save_path, 'boundary')
        boundary_path.mkdir(parents=True, exist_ok=True)
        xcx_path = Path(save_path, 'xcx')
        xcx_path.mkdir(parents=True, exist_ok=True)
        
        print(image_path.name)
        plt.figure(figsize=(20, 20))
        plt.subplot(2, 3, 1), plt.imshow(img, cmap="gray"), plt.title("img"), plt.axis("off")
        plt.subplot(2, 3, 2), plt.imshow(label_xcx, cmap="gray"), plt.title("label_xcx"), plt.axis("off") # label xcx
        plt.subplot(2, 3, 3), plt.imshow(xcx_img, cmap="gray"), plt.title("output_xcx"), plt.axis("off")
        plt.subplot(2, 3, 4), plt.imshow(label_boundary, cmap="gray"), plt.title("label_boundary"), plt.axis("off")
        plt.subplot(2, 3, 5), plt.imshow(classical_out_img, cmap="gray"), plt.title("classical_post_out_img"), plt.axis("off")
        plt.subplot(2, 3, 6), plt.imshow(water_out_img, cmap="gray"), plt.title("ours_out_img"), plt.axis("off")
        plt.show()

        # 把推理结果存下来看看。

        # cv2.imwrite('./post2/FESEM/boundary/'+label_name,)
        cv2.imwrite(str(Path(boundary_path, label_name)), out_img)
        cv2.imwrite(str(Path(xcx_path, label_name)), xcx_img)




if __name__ == '__main__':
    mode = "FESEM" # OM
    weight_phta_path = './test-pth/'+mode+'_best_model_state.pth'
    infer_path = './data/'+mode+'/test_img'
    save_path = './infer/'+mode+'/'

    use_post_process = True
    inference(weight_phta_path, infer_path, save_path, use_post_process)