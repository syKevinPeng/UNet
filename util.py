import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import tensorboard as tb
msrc_directory = 'SegmentationDataset'
DEVICE = 'cuda:0'

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
    plt.show()


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

def batch_accuracy(preds:torch.Tensor,gts:torch.Tensor, threshold = 0.5):
    preds = preds.detach().clone()
    if preds.shape != gts.shape:
        raise Exception("Wrong prediction and Ground Truth shape")
    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0
    total = gts.nelement()
    acc = (total - torch.count_nonzero(preds-gts))/total
    return acc.item()

    # gt, pred = torch.flatten(gt), torch.flatten(pred)
    # gt = gt.cpu().detach().numpy()
    # pred = pred.cpu().detach().numpy()
    # pred[pred > threshold] = 1
    # pred[pred <= threshold] = 0
    # acc = (len(gt) - np.count_nonzero(pred-gt))/(len(gt))
    # return acc

def dataset_stats():
    img_list_train_val = [x.split('.')[-2].split('/')[-1][:-3] for x in glob.glob(msrc_directory + '/train/*')
                               if 'GT' in x]
    dataset_name = ['%s/%s.bmp' % (msrc_directory, x) for x in img_list_train_val]
    dataset = torch.as_tensor([])
    for img_name in dataset_name:
        # print(f'img name {img_name}')
        img = torch.as_tensor(np.array(plt.imread(img_name)))
        if len(dataset) == 0:
            dataset = torch.unsqueeze(img, 0)
        else:
            if img.shape != torch.Size([213,320,3]):
                img = torch.swapaxes(img, 0, 1)
            # print(f'img shape: {torch.unsqueeze(img, 0).shape}')
            dataset = torch.cat((dataset,torch.unsqueeze(img, 0)))
    print(f'Calculating Mean and Std of the Dataset ...')
    dataset = dataset.to(DEVICE).float()
    imgs_mean = torch.mean(dataset,dim=(0,1,2))
    imgs_std = torch.std(dataset,dim=(0,1,2))
    print(f'imgs mean: {imgs_mean}\nimgs std: {imgs_std}')
    return imgs_mean, imgs_std

def show_test_result(result_path):
    masks = torch.load(result_path)
    print(f'masks: ', masks)
    assert (masks.shape == (24, 256, 256))
    assert ((torch.where(masks == 1, 10, 0).sum() + torch.where(masks == 0, 10, 0).sum()).item() == 24 * 256 * 256 * 10)
    masks = torch.moveaxis(masks, 0, 2)
    rand_idx = np.random.randint(0, 23)
    rand_mask = masks[:,:,rand_idx].cpu().detach().numpy()
    plt.imshow(rand_mask)
    plt.title(str(rand_idx))
    plt.show()

def plot_train_val_acc_loss(experiment_id, run_name):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    best_run = df.loc[df['run'] == run_name]
    loss_train = best_run.loc[best_run['tag'] == 'Loss/train']
    loss_val = best_run.loc[best_run['tag'] == 'Loss/validation']
    acc_train = best_run.loc[best_run['tag'] == 'Accuracy/train']
    acc_val = best_run.loc[best_run['tag'] == 'Accuracy/validation']
    loss_batch_train = best_run.loc[best_run['tag'] == 'Loss_batch/train']
    loss_batch_val = best_run.loc[best_run['tag'] == 'Loss_batch/validation']
    print(loss_batch_train)


    plt.figure(figsize=(16, 6))
    # Plot training and validation loss per batch
    plt.subplot(2, 2, 1)
    plt.plot(loss_batch_train['step'], loss_batch_train['value'])
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.title('Training loss per batch')
    plt.subplot(2, 2, 2)
    plt.plot(loss_batch_val['step'], loss_batch_val['value'])
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.ylim((0.6, 4))
    plt.title('Validation loss per batch')
    # plot training and validation accuracy per epoch
    plt.subplot(2, 2, 3)
    plt.plot(acc_train['step'], acc_train['value'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.subplot(2, 2, 4)
    plt.plot(acc_val['step'], acc_val['value'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Validation accuracy per Epoch')
    plt.show()

if __name__ == "__main__":
    plot_train_val_acc_loss(experiment_id="tQEQnsA4S2KY6znAqP8IGQ", run_name="Apr13_16-59-07_Tuge-PCLR_0.07_BS_32")