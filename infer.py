#%%
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
from post_proc import boundary_proc, xcx_proc, save_image


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
    :param weight_path:  权重路径
    :param root_path:  想要推理的文件根目录
    :param save_path:  推理结果保存目录
    :param proc_method:  处理方式，选择 boundary or xcx
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
        if b == '.pth':  # pth文件load
            net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight_path).items()})
        elif b == '.params':  # params文件load
            net.load_state_dict(torch.load(weight_path))
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

        out_tensor, xcx_tensor = net(image)
        out_tensor = F.softmax(out_tensor, dim = 1)
        xcx_tensor = F.softmax(xcx_tensor, dim = 1)

        out_tensor = out_tensor[0, :, :, :].cpu().detach().numpy().transpose(1,2,0)
        xcx_tensor = xcx_tensor[0, :, :, :].cpu().detach().numpy().transpose(1,2,0)

        out_img = np.argmax(out_tensor, axis=2).astype(np.uint8)
        xcx_img = np.argmax(xcx_tensor, axis=2).astype(np.uint8)
        if use_post_process:
            out_img = boundary_proc(out_img)
            xcx_img = xcx_proc(xcx_img)

        out_img[out_img>0] = 255
        xcx_img[xcx_img>0] = 255

        
        label_name = image_path.name
        boundary_path = Path(save_path, 'boundary')
        boundary_path.mkdir(parents=True, exist_ok=True)
        xcx_path = Path(save_path, 'xcx')
        xcx_path.mkdir(parents=True, exist_ok=True)
        
        print(image_path.name)
        plt.figure(figsize=(20, 20))
        plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title("img"), plt.axis("off")
        plt.subplot(1, 3, 2), plt.imshow(out_img, cmap="gray"), plt.title("output_boundary"), plt.axis("off")
        plt.subplot(1, 3, 3), plt.imshow(xcx_img, cmap="gray"), plt.title("output_xcx"), plt.axis("off")
        plt.show()

        cv2.imwrite(str(Path(boundary_path, label_name)), out_img)
        cv2.imwrite(str(Path(xcx_path, label_name)), xcx_img)




if __name__ == '__main__':
    weight_phta_path = '/root/data/zhangxinyi/data/gyy-state2-train/test-pth/OM_best_model_state.pth'
    infer_path = '/root/data/zhangxinyi/data/gyy-state2-train/data/OM/test_img'
    save_path = '/root/data/zhangxinyi/data/gyy-state2-train/infer/OM/'

    # 晶界推理
    use_post_process = False
    inference(weight_phta_path, infer_path, save_path, use_post_process)
# %%
