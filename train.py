import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.utils import data
from tqdm import tqdm
import numpy as np
from models import RDN
from datasets import MyDataset
from utils import AverageMeter, calc_psnr, convert_rgb_to_y, denormalize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=800)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = RDN(scale_factor=args.scale,
                # num_channels=3,
                num_channels=1,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    if args.weights_file is not None:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    img_path ='/Users/apple/Downloads/image-super-resolution-master-2/notebooks/div2k/DIV2K_train_LR_bicubic/' #'/media/cmu/DATA/francis/Melt_Pool_Data/superresdata_text/10filterresdata_cropped_resized_topdown/'#'/media/cmu/DATA/francis/Melt_Pool_Data/superresdata_text/lowresdata_cropped_resized_topdown/'#'/media/cmu/DATA/francis/Melt_Pool_Data/superresdata_text/lowresdata_cropped_filtered_size_ten/'#'/home/cmu-mail/Desktop/Bode_Fujah/Research/Super_Resolution/HR_LR_Files/LR_BoxFilterSize10/lowresdata_cropped_filtered_size_ten/'
    targ_path = '/Users/apple/Downloads/image-super-resolution-master-2/notebooks/div2k/DIV2K_train_HR/'#'/media/cmu/DATA/francis/Melt_Pool_Data/superresdata_text/highresdata_cropped_fixedsizing/'#'/home/cmu-mail/Desktop/Bode_Fujah/Research/Super_Resolution/HR_LR_Files/HR_BoxFilterSize10/highresdata_cropped_fixedsizing/'
    img_names = os.listdir(img_path)
    targ_names = os.listdir(targ_path)
    intersection_list = list(set(img_names) & set(targ_names))
    all_paths = np.array([img_path+img_name for img_name in  intersection_list if img_name.endswith('npy')],dtype = 'object')
    targ_paths = np.array([targ_path+img_name for img_name in  intersection_list if img_name.endswith('npy')],dtype = 'object')
    index_array = np.arange(len(all_paths)) 
    np.random.shuffle(index_array)
    train  = index_array[:int(len(all_paths)*0.75)]
    dev  = index_array[int(len(all_paths)*0.75):int(len(all_paths)*0.85)]
    test = index_array[int(len(all_paths)*0.85):]
    # print(train)

    image_paths = all_paths[train]#[img_path+img_name for img_name in (os.listdir(img_path)) if img_name.endswith('npy')][:3500]#["DNS-LES_3C/les_3c/"+str(i/5670)[:3]+"0/"+str(i%567)+".png" for i in range(nums)]
    target_paths = targ_paths[train]##["DNS-LES_3C/dns_3c/"+str(i/5670)[:3]+"0/"+str(i%567)+".png" for i in range(nums)] 
    # print(len(image_paths))
    # print(len(target_paths))
    image_paths_dev = all_paths[dev] #['/media/cmu/DATA/francis/Melt_Pool_Data/superresdata/lowresdata_resized/' + img_name for img_name in (os.listdir('/media/cmu/DATA/francis/Melt_Pool_Data/superresdata/lowresdata_resized' )) if img_name.endswith('png')]#["DNS-LES_3C/les_3c/"+str(i/5670)[:3]+"0/"+str(i%567)+".png" for i in range(10000, 10000+nums)]
    target_paths_dev =targ_paths[dev]#['/medi
    image_paths_test =all_paths[test] #[img_path+img_name for img_name in (os.listdir(img_path)) if img_name.endswith('npy')][3500:]#['/media/cmu/DATA/francis/Melt_Pool_Data/superresdata/lowresdata_resized/' + img_name for img_name in (os.listdir('/media/cmu/DATA/francis/Melt_Pool_Data/superresdata/lowresdata_cropped')) if img_name.endswith('png')]#["DNS-LES_3C/les_3c/"+str(i/5670)[:3]+"0/"+str(i%567)+".png" for i in range(20000, 20000+nums)]
    target_paths_test = targ_paths[test]#

    batch_size=args.batch_size
    train_dataset = MyDataset(image_paths, target_paths)
    train_loader_args = dict(batch_size=batch_size, shuffle=True, num_workers=8)
    train_loader = data.DataLoader(train_dataset, **train_loader_args)
    # print(next(iter(train_loader)))
    dev_dataset = MyDataset(image_paths_dev, target_paths_dev)
    dev_loader_args = dict(batch_size=batch_size, shuffle=False, num_workers=8)
    dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)
    # print(dev_dataset.__len__())

    test_dataset = MyDataset(image_paths_test, target_paths_test)
    test_loader_args = dict(batch_size=1, shuffle=False, num_workers=8)
    test_loader = data.DataLoader(test_dataset, **test_loader_args)
    # print(test_dataset.__len__())

    # train_dataset = TrainDataset(args.train_file, patch_size=args.patch_size, scale=args.scale)
    # train_dataloader = DataLoader(dataset=train_dataset,
    #                               batch_size=args.batch_size,
    #                               shuffle=True,
    #                               num_workers=args.num_workers,
    #                               pin_memory=True)
    # eval_dataset = EvalDataset(args.eval_file)
    # eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    loss_list = []
    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_loader:
                inputs, labels = data
                # Reshape inputs:
                batch_size_inputs = inputs.shape[0]
                lenth_inputs = inputs.shape[1]
                width_inputs = inputs.shape[2]
                inputs = np.reshape(inputs, (batch_size_inputs, 1, lenth_inputs, width_inputs))
                # Reshape labels:
                batch_size_labels = labels.shape[0]
                lenth_labels = labels.shape[1]
                width_labels = labels.shape[2]
                labels = np.reshape(labels, (batch_size_labels, 1, lenth_labels, width_labels))

                inputs = inputs.to(device)
                labels = labels.to(device)
        
                preds = model(inputs, epoch = epoch)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        loss_list.append(epoch_losses.avg)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in dev_loader:
            inputs, labels = data
            # Reshape inputs:
            batch_size_inputs = inputs.shape[0]
            lenth_inputs = inputs.shape[1]
            width_inputs = inputs.shape[2]
            inputs = np.reshape(inputs, (batch_size_inputs, 1, lenth_inputs, width_inputs))
            # Reshape labels:
            batch_size_labels = labels.shape[0]
            lenth_labels = labels.shape[1]
            width_labels = labels.shape[2]
            labels = np.reshape(labels, (batch_size_labels, 1, lenth_labels, width_labels))
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)

            preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
            labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

            preds = preds[args.scale:-args.scale, args.scale:-args.scale]
            labels = labels[args.scale:-args.scale, args.scale:-args.scale]

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
    loss_list = np.array(loss_list)
    np.savetxt('/Users/apple/Downloads/RDN-pytorch-master/whole_loss', loss_list)
