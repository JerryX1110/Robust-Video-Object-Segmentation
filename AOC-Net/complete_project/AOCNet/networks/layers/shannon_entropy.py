import torch
import matplotlib.pyplot as plt
import numpy as np

def normalize(image, MIN_BOUND, MAX_BOUND):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    reverse_image = 1 - image
    return reverse_image

def cal_shannon_entropy(preds):  # (batch, obj_num, 128, 128)
    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)  # (batch, 1, 128, 128)
    uncertainty_norm = normalize(uncertainty, 0, np.log(2)) * 7
    return uncertainty,uncertainty_norm


def normalize_train(image, MIN_BOUND, MAX_BOUND):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    #reverse_image = 1 - image
    return image #reverse_image

def cal_shannon_entropy_train(preds):  # (batch, obj_num, 128, 128)
    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)  # (batch, 1, 128, 128)
    uncertainty_norm = normalize_train(uncertainty, 0, np.log(2)) * 7
    return uncertainty,uncertainty_norm



def show_shannon_entropy(uncertainty,uncertainty_norm,unc_rate=0.5):   #torch.tensor, torch.tensor
    uncertainty = uncertainty.cpu().numpy().squeeze().astype('float32')
    uncertainty = uncertainty * (uncertainty > unc_rate)
    uncertainty_norm = uncertainty_norm.cpu().numpy().squeeze().astype('float32')
    uncertainty_norm = uncertainty_norm

    plt.figure()
    plt.subplot(1, 6, 1)
    plt.imshow(save_pre_den)
    plt.subplot(1, 6, 2)
    plt.imshow(density_gt)
    plt.subplot(1, 6, 3)
    plt.imshow(save_pre_dmp_to_att)
    plt.subplot(1, 6, 4)
    plt.imshow(save_pre_att_2)
    plt.subplot(1, 6, 5)
    plt.imshow(uncertainty, cmap='inferno')
    plt.subplot(1, 6, 6)
    plt.imshow(uncertainty_norm, cmap='inferno')
    plt.show()

def save_shannon_entropy(uncertainty,uncertainty_norm,save_path,unc_rate=0.5):   #torch.tensor, torch.tensor
    uncertainty = uncertainty.cpu().numpy().squeeze().astype('float32')
    uncertainty = uncertainty * (uncertainty > unc_rate)
    uncertainty_norm = uncertainty_norm.cpu().numpy().squeeze().astype('float32')
    uncertainty_norm = uncertainty_norm

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(uncertainty, cmap='inferno')
    plt.subplot(1, 2, 2)
    plt.imshow(uncertainty_norm, cmap='inferno')
    #plt.show()
    plt.savefig(save_path)
    plt.close()

def save_shannon_entropy_calculated(uncertainty,save_path,unc_rate=1):   #torch.tensor, torch.tensor
    uncertainty = uncertainty.cpu().numpy().squeeze().astype('float32')

    

    plt.figure()
    unc_rate=0
    uncertainty_org1 = uncertainty
    uncertainty_thresh1 =  (uncertainty > unc_rate)
    uncertainty1 = uncertainty * (uncertainty > unc_rate)
    plt.subplot(3,3, 1)
    plt.imshow(uncertainty_org1, cmap='inferno')
    plt.subplot(3,3, 2)
    plt.imshow(uncertainty_thresh1, cmap='inferno')
    plt.subplot(3,3, 3)
    plt.imshow(uncertainty1, cmap='inferno')

    unc_rate=1
    uncertainty_org2 = uncertainty
    uncertainty_thresh2 =  (uncertainty > unc_rate)
    uncertainty2 = uncertainty * (uncertainty > unc_rate)
    plt.subplot(3,3, 4)
    plt.imshow(uncertainty_org2, cmap='inferno')
    plt.subplot(3,3, 5)
    plt.imshow(uncertainty_thresh2, cmap='inferno')
    plt.subplot(3,3, 6)
    plt.imshow(uncertainty2, cmap='inferno')

    unc_rate=10
    uncertainty_org3 = uncertainty
    uncertainty_thresh3 =  (uncertainty > unc_rate)
    uncertainty3 = uncertainty * (uncertainty > unc_rate)
    plt.subplot(3,3, 7)
    plt.imshow(uncertainty_org2, cmap='inferno')
    plt.subplot(3,3, 8)
    plt.imshow(uncertainty_thresh2, cmap='inferno')
    plt.subplot(3,3, 9)
    plt.imshow(uncertainty2, cmap='inferno')



    #plt.show()
    plt.savefig(save_path)
    plt.close()