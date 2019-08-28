from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import data_manager
from video_loader import VideoDataset
import transforms as T
import models
from models import resnet3d
from losses import CrossEntropyLabelSmooth, TripletLoss
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from tqdm import tqdm
from opt import args
from pathlib import Path
import pandas as pd

import gc

from torch import multiprocessing as mp

def main():
    mp.set_start_method('spawn', True)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    if args.dataset == 'mars':
        num_train_pids = 625
    elif args.dataset == 'viva':
        num_train_pids = 204

    ## Initializing dataset
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = False # pin_memory = True if use_gpu else False

    if not args.simi:
        print("Initializing dataset {}".format(args.dataset))
        dataset = data_manager.init_dataset(name=args.dataset)

        trainloader = DataLoader(
            VideoDataset(dataset.train, seq_len=args.seq_len, sample='random',transform=transform_train),
            sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
            batch_size=args.train_batch, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=True,
        )

        queryloader = DataLoader(
            VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

        galleryloader = DataLoader(
            VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

    print("Initializing model: {}".format(args.arch))
    if args.arch=='resnet503d':
        model = resnet3d.resnet50(num_classes=num_train_pids, sample_width=args.width, sample_height=args.height, sample_duration=args.seq_len)
        if not os.path.exists(args.pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
        print("Loading checkpoint from '{}'".format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
        state_dict = {}
        for key in checkpoint['state_dict']:
            if 'fc' in key: continue
            state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict, strict=False)
    else:
        model = models.init_model(name=args.arch, num_classes=num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_xent = CrossEntropyLabelSmooth(num_classes=num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        resume_filename = osp.join(args.save_dir, 'best_model.pth.tar')
        checkpoint = torch.load(resume_filename, map_location=torch.device('cuda' if use_gpu else 'cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        if use_gpu: model = nn.DataParallel(model).cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_rank1 = checkpoint['rank1']
        start_epoch = checkpoint['epoch'] + 1
        print("Loading from Epoch {}".format(start_epoch))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    
    if args.simi:
        print('Calculating similarity scores only')
        simi(model, args, transform_test, use_gpu)
        return

    if args.evaluate:
        print("Evaluate only")
        test(model, queryloader, galleryloader, args.pool, use_gpu)
        return

    start_time = time.time()
    best_rank1 = -np.inf
    if args.arch=='resnet503d':
        torch.backends.cudnn.benchmark = False
    
    for epoch in range(start_epoch, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        
        train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)
        
        if args.stepsize > 0: scheduler.step()
        
        if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch or epoch == 1:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, args.pool, use_gpu)
            is_best = rank1 > best_rank1
            if is_best: best_rank1 = rank1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def extract_features(model, data_loader, use_gpu, pool='avg'):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(tqdm(data_loader)):
        b, n, s, c, h, w = imgs.size()
        assert(b==1)
        imgs = imgs.view(b*n, s, c, h, w)
        with torch.no_grad():
            if n > 200:
                cuts = get_cuts(n, 200)
                features = list()
                for i in range(len(cuts)-1):
                    img = imgs[cuts[i]:cuts[i+1]]
                    if use_gpu: img = img.cuda()
                    features.append(model(img))
                features = torch.cat(features, dim=0)
            else:
                if use_gpu: imgs = imgs.cuda()
                features = model(imgs)

        features = features.view(n, -1)
        if pool == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        features = features.data.cpu()
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
        del features, imgs, pids, camids
        gc.collect()
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
    return qf, q_pids, q_camids

def get_cuts(n, max_seq_len=900):
    cuts = n // max_seq_len # 8104 // 800 = 10           8000 // 800 = 10
    cuts = [i*max_seq_len for i in range(cuts)] # 0 800 1600 .. 7200
    cuts.append(len(cuts)*max_seq_len) # 0 800 1600 .. 7200 8000
    if n > cuts[-1]: cuts.append(n) # 0 800 1600 .. 7200 8000 8104
    return cuts

def train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()

    for batch_idx, (imgs, pids, _) in enumerate(tqdm(trainloader)):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        imgs, pids = Variable(imgs), Variable(pids)
        outputs, features = model(imgs)
        if args.htri_only:
            # only use hard triplet loss to train the network
            loss = criterion_htri(features, pids)
        else:
            # combine hard triplet loss with cross entropy loss
            xent_loss = criterion_xent(outputs, pids)
            htri_loss = criterion_htri(features, pids)
            loss = xent_loss + htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))

def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()

    qf, q_pids, q_camids = extract_features(model, queryloader, use_gpu, pool=pool)
    gf, g_pids, g_camids = extract_features(model, galleryloader, use_gpu, pool=pool)
    del queryloader, galleryloader
    gc.collect()

    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    del qf, gf 
    gc.collect()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]

def simi(model, args, transform_test, use_gpu):
    root = args.path
    model.eval()

    root_path_len = len(Path(root).parts)
    tpaths = [Path(tpath) for tpath, _, __ in os.walk(root)]
    tpaths = [tpath for tpath in tpaths if len(tpath.parts) - 2 == root_path_len]

    tracklets = []
    for tpath in tpaths:
        img_paths = list(Path(tpath).glob('**/*.[pj][np]g'))
        tracklets.append((tuple(img_paths), 0, 0))
    loader = DataLoader(
        VideoDataset(tracklets, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=False, drop_last=False,
    )

    f, _pids, _camids = extract_features(model, loader, use_gpu)
    del loader, model
    gc.collect()
    n = f.size(0)
    tmp = torch.pow(f, 2).sum(dim=1, keepdim=True).expand(n, n)
    distmat = tmp + tmp.t()
    distmat.addmm_(1, -2, f, f.t())
    distmat = distmat.numpy()
    del f
    gc.collect()

    t_names = [tpath.parts[-2]+'/'+tpath.parts[-1] for tpath in tpaths]
    distdf = pd.DataFrame(distmat, columns=t_names, index=t_names)
    distdf.to_csv('result.csv')

if __name__ == '__main__':
    main()