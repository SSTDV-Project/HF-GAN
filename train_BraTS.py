from data.dataset import *
from utils import *
from network.network import *

import random, argparse, time, itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from accelerate import Accelerator
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure

def set_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='BraTS')

    parser.add_argument('--dataset', type=str, required=True,
                        help='path of training dataset')
    parser.add_argument('--identifier', type=str, required=True, metavar='N',
                        help='Select the identifier for file name')
    parser.add_argument('--batch-size', type=int,  default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--ch-dim', type=int,  default=64, metavar='N',
                        help='channel dimension for netwrok (default: 64)')
    parser.add_argument('--gradient_accumulation_steps', type=int,  default=1, metavar='N',
                        help='gradient_accumulation_steps for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epoches to train (default: 100)')
    parser.add_argument('--numlayers', type=int, default=4, metavar='N',
                        help='number of transformer layers(default: 4)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='path of training snapshot')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--gpu', type=str, default='0', metavar='N',
                        help='Select the GPU (defualt 0)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='number of epoches to log (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--resume', action='store_true',
                        help='resume training by loading last snapshot')
    args = parser.parse_args()

    set_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    
    train_dataset = BraTSDataset(args.dataset, mode='train')
    valid_dataset = BraTSDataset(args.dataset, mode='valid')    
    
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, generator=generator)
    
    model = HFGAN(dim=args.ch_dim, num_inputs=4, num_outputs=1, dim_mults=(1,2,4,8,10), n_layers=args.numlayers, skip=True, blocks=False)
    discriminator = Discriminator(channels=1, num_filters_last=args.ch_dim)

    optimizer = Adam(model.parameters(), lr=0.0)
    optimizer_D = Adam(discriminator.parameters(), lr=0.0)
    steps_per_epoch = len(train_loader)
    total_iteration = epochs*steps_per_epoch*4
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=total_iteration, T_mult=1, eta_max=args.lr, T_up=100, gamma=0.5)
    scheduler_D = CosineAnnealingWarmUpRestarts(optimizer_D, T_0=total_iteration, T_mult=1, eta_max=args.lr*0.1, T_up=100, gamma=0.5)

    accelerator = Accelerator(gradient_accumulation_steps = args.gradient_accumulation_steps,)
    
    device = accelerator.device
    
    valid_epochs = args.log_interval
    
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    discriminator, optimizer_D, scheduler_D = accelerator.prepare(
        discriminator, optimizer_D, scheduler_D)
    print(f'Total Iteration: {total_iteration}')
    
    epoch = 0
    iterations = 0
    if args.resume:
        accelerator.load_state(input_dir=os.path.join(args.checkpoints,args.identifier))
        iterations = scheduler.scheduler.T_cur
        epoch = scheduler.scheduler.T_cur // steps_per_epoch / 4

    print(f'iteration: {iterations} epoch : {epoch}')
    
    loss_adversarial = torch.nn.BCEWithLogitsLoss()
    loss_auxiliary = torch.nn.CrossEntropyLoss()
    metric_psnr = PeakSignalNoiseRatio(data_range = 1.0).to(device)
    metric_siim = StructuralSimilarityIndexMeasure(data_range = 1.0).to(device)
    metric_mssiim = MultiScaleStructuralSimilarityIndexMeasure(data_range = 1.0).to(device)
    
    cand = [0, 1, 2, 3]
    candidates_all = []
    for L in range(len(cand) + 1):
        if L == 0 or L == 1:
            continue
        for subset in itertools.combinations(cand, L):
            candidates_all.append(subset)
    candidates = [list(filter(lambda x:m not in x, candidates_all)) for m in cand]
    while epoch < epochs:
        epoch +=1
        avg_train_total_loss = []
        avg_train_adversarial_loss = []
        avg_train_auxiliary_loss = []
        avg_train_d_real_loss = []
        avg_train_d_fake_loss = []

        for n, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                model.train()
                discriminator.train()
                
                inputs_all = batch['image'] # BxCxWxH
                targets = batch['target']
                modalities = batch['modalitiy'].squeeze(dim=-1)

                for m_shift in [False, True, True, True]:
                    iterations += 1
                    if m_shift:
                        for idx in range(inputs_all.shape[0]):
                            modalities[idx] = modalities[idx]+1 if modalities[idx]!=3 else 0
                            targets[idx,:] = inputs_all[idx,modalities[idx]:modalities[idx]+1,:]

                    targets_second = torch.zeros_like(targets, device=targets.device)
                    inputs_masked = -1*torch.ones_like(inputs_all, device=targets.device)
                    inputs_masked_second = -1*torch.ones_like(inputs_all, device=targets.device)
                    modalities_second = torch.zeros_like(modalities, device=modalities.device)

                    input_modals = []
                    input_modals2 = []
                    for n, m in enumerate(modalities):
                        if n < inputs_all.shape[0] // 2: 
                            cand = random.choice(candidates[m])

                            input_modals.append(cand)
                            for c in cand:
                                inputs_masked[n,c,:] = inputs_all[n,c,:]
                            m_masked = random.choice(cand)
                            modalities_second[n] = m_masked

                            cand2 = [x for x in [0,1,2,3] if x != m_masked]
                            input_modals2.append(cand2)
                            for c in cand2:
                                inputs_masked_second[n,c,:] = inputs_all[n,c,:]
                            targets_second[n,:] = inputs_all[n,m_masked:m_masked+1,:]
                        else:
                            cand = random.choice([x for x in [0,1,2,3] if x != m])
                            input_modals.append([cand])
                            inputs_masked[n,cand,:] = inputs_all[n,cand,:]
                            modalities_second[n] = cand
                            
                            cand2 = m
                            input_modals2.append([cand2])
                            inputs_masked_second[n,cand2,:] = inputs_all[n,cand2,:]
                            targets_second[n,:] = inputs_all[n,cand:cand+1,:]

                    #train G
                    optimizer.zero_grad()

                    f, h = model.encoder(inputs_masked, input_modals, train_mode=True)
                    z = model.middle(f, modalities)
                    targets_recon = model.decoder(z, h)

                    recon_loss = (torch.abs(targets - targets_recon)).mean()

                    for n, m in enumerate(modalities):
                        inputs_masked_second[n,m,:] = targets_recon[n,0,:]
                    f_recon, h_recon = model.encoder(inputs_masked_second, input_modals2, train_mode=True)
                    feature_l1_loss = 1 - F.cosine_similarity(f.flatten(1,-1), f_recon.flatten(1,-1)).mean()
                    
                    z_recon = model.middle(f_recon, modalities_second)

                    targets_cycle = model.decoder(z_recon, h_recon)
                    cycle_loss = (torch.abs(targets_second - targets_cycle)).mean()

                    logits_fake, labels_fake = discriminator(targets_recon)
                
                    valid=torch.ones(logits_fake.shape).cuda()
                    fake=torch.zeros(logits_fake.shape).cuda()

                    adversarial_loss = loss_adversarial(logits_fake, valid)
                    auxiliary_loss = loss_auxiliary(labels_fake, modalities)

                    total_loss = 10*recon_loss +0.25*adversarial_loss + 0.25*auxiliary_loss + 1*feature_l1_loss + 1*cycle_loss

                    accelerator.backward(total_loss)
                    optimizer.step()

                    optimizer_D.zero_grad()

                    logits_real, labels_real = discriminator(targets_second)
                    logits_fake, labels_fake = discriminator(targets_recon.detach())
                    
                    d_real_adv = loss_adversarial(logits_real, valid)
                    d_fake_adv = loss_adversarial(logits_fake, fake)
                    
                    d_real_aux = loss_auxiliary(labels_real, modalities_second)
                    d_fake_aux = loss_auxiliary(labels_fake, modalities)

                    d_loss = 0.25*(d_real_adv + d_fake_adv) + 0.25*d_real_aux + 0.25*d_fake_aux
                    accelerator.backward(d_loss)
                    
                    optimizer_D.step()

                    scheduler.step(iterations)
                    scheduler_D.step(iterations)
                    
                    avg_train_total_loss.append(total_loss.item())
                    avg_train_adversarial_loss.append(adversarial_loss.item())
                    avg_train_auxiliary_loss.append(auxiliary_loss.item())
                    avg_train_d_real_loss.append(d_real_adv.item())
                    avg_train_d_fake_loss.append(d_fake_adv.item())


        if accelerator.is_main_process:
            print(f"Train Loss: {np.mean(avg_train_total_loss):.6f}, G Loss: {np.mean(avg_train_adversarial_loss)+np.mean(avg_train_auxiliary_loss):.6f}, D Loss: {np.mean(avg_train_d_real_loss)+np.mean(avg_train_d_fake_loss):.6f}")

        if epoch % valid_epochs == 0:
            accelerator.save_state(output_dir=os.path.join(args.checkpoints,args.identifier))
            with torch.no_grad():
                avg_valid_recon_loss = []
                avg_valid_psnr = []
                avg_valid_ssim = []
                avg_valid_msssim = []
                for batch in valid_loader:
                    model.eval()
                    
                    inputs = batch['image_masked'] # BxCxWxH
                    targets = batch['target']
                    modalities = batch['modalitiy'].squeeze(dim=-1)
                    recon_list = []
                    input_modals = []
                    for n, m in enumerate(modalities):
                        input_modals.append([x for x in [0,1,2,3] if x != m])

                    f, h = model.encoder(inputs, input_modals)
                    z = model.middle(f, modalities)
                    targets_recon = model.decoder(z, h)
                    recon_list.append(targets_recon.cpu().numpy())
                    for j in range(targets_recon.shape[0]):
                        avg_valid_recon_loss.append(torch.abs(targets[j:j+1,:] - targets_recon[j:j+1,:]).mean().cpu())
                        avg_valid_psnr.append(metric_psnr(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu())
                        avg_valid_ssim.append(metric_siim(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu())
                        avg_valid_msssim.append(metric_mssiim(targets_recon[j:j+1,:], targets[j:j+1,:]).cpu())
                if accelerator.is_main_process:
                    print(f"Valid Recon Loss: {np.mean(avg_valid_recon_loss):.6f}, PSNR: {np.mean(avg_valid_psnr):.6f}, , SSIM: {np.mean(avg_valid_ssim):.6f}, , MS-SSIM: {np.mean(avg_valid_msssim):.6f}")
                        

if __name__ == "__main__":
    main()
    