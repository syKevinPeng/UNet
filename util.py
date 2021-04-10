import numpy as np
import matplotlib.pyplot as plt


SEG_LABELS_LIST_v1 = [
    {"id": -1, "name": "void",       "rgb_values": [0,   0,    0]},
    {"id": 0,  "name": "building",   "rgb_values": [128, 0,    0]},
    {"id": 1,  "name": "grass",      "rgb_values": [0,   128,  0]},
    {"id": 2,  "name": "tree",       "rgb_values": [128, 128,  0]},
    {"id": 3,  "name": "cow",        "rgb_values": [0,   0,    128]},
    {"id": 4,  "name": "sky",        "rgb_values": [128, 128,  128]},
    {"id": 5,  "name": "airplane",   "rgb_values": [192, 0,    0]},
    {"id": 6, "name": "face",       "rgb_values": [192, 128,  0]},
    {"id": 7, "name": "car",        "rgb_values": [64,  0,    128]},
    {"id": 8, "name": "bicycle",    "rgb_values": [192, 0,    128]}]

background_classes=["void","grass","sky"]
background_colors=[]
for i in range(len(SEG_LABELS_LIST_v1)):
    if SEG_LABELS_LIST_v1[i]["name"] in background_classes:
        background_colors.append(SEG_LABELS_LIST_v1[i]["rgb_values"])

def get_binary_seg(bgr_seg):
    rgb_seg = bgr_seg  # [:,:,::-1]#reverse order of channels from bgr to rgb
    shape_rgb = rgb_seg.shape
    binary_shape = (shape_rgb[0], shape_rgb[1], 1)

    binary_map = np.ones(binary_shape)
    for background_color in background_colors:
        binary_map[(rgb_seg == background_color).all(2)] = 0

    return binary_map

def plot_image(im,title,xticks=[],yticks= [],cv2 = True):
    """
    im :Image to plot
    title : Title of image
    xticks : List of tick values. Defaults to nothing
    yticks :List of tick values. Defaults to nothing
    cv2 :Is the image cv2 image? cv2 images are BGR instead of RGB. Default True
    """

    plt.figure()
    if(im.shape[2]==1):
        plt.imshow(np.squeeze(im),cmap='gray')
    elif cv2:
        plt.imshow(im[:,:,::-1])
    else:
        plt.imshow(im)
    plt.title(title)
    plt.xticks(xticks)
    plt.yticks(yticks)

def dataloader_tester(train_dataloader, val_dataloader, test_dataloader):
    input, labels = next(iter(train_dataloader))
    print(input.shape, labels.shape)
    print(type(input[2]))
    img = input[2].numpy().transpose(1, 2, 0)
    mask = labels[2].numpy().transpose(1, 2, 0)
    plot_image(img, 'train_image', cv2=False)
    plot_image(mask, 'train_seg')

    input, labels = next(iter(val_dataloader))
    print(input.shape, labels.shape)
    img = input[2].numpy().transpose(1, 2, 0)
    mask = labels[2].numpy().transpose(1, 2, 0)
    plot_image(img, 'val_image', cv2=False)
    plot_image(mask, 'val_seg')

    input = next(iter(test_dataloader))
    print(input.shape)
    img = input[2].numpy().transpose(1, 2, 0)
    plot_image(img, 'test_image', cv2=False)