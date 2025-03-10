from mmaction.apis import init_recognizer
import torch
import argparse
from tqdm import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from dataloader_video_flow_epic import EPICDOMAIN

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
from torch.nn.parameter import Parameter

default_net_type = 'oneM+bias'
default_Enc = False


class HyperNetwork(nn.Module):
    # Kin 2048 Try
    def __init__(self, hidden=3072, z_dim=4352, label_size=1501, batch_size=32, zoom_dim=4352,
                 net_type=default_net_type):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.hidden = hidden
        self.label_size = label_size
        self.batch_size = batch_size
        self.zoom_dim = zoom_dim
        self.net_type = net_type

        if self.net_type == 'twoM+bias':
            self.w1 = Parameter(torch.randn((self.hidden, self.zoom_dim * self.label_size)).cuda())  # 128, 96*1001
            self.b1 = Parameter(torch.randn((self.label_size * self.zoom_dim)).cuda())

            self.w2 = Parameter(torch.randn((self.z_dim, self.hidden)).cuda())  # 192,128
            self.b2 = Parameter(torch.randn((self.hidden)).cuda())

            self.b_hyper1 = Parameter(torch.randn(self.hidden, self.label_size)).cuda()

            self.bn_h_in = nn.BatchNorm1d(self.hidden).cuda()
            self.bn_kernel = nn.BatchNorm1d(self.zoom_dim * self.label_size).cuda()
            self.bn_bias = nn.BatchNorm1d(self.label_size).cuda()

        if self.net_type == 'oneM+bias':
            self.w3 = Parameter(torch.randn((self.z_dim, self.zoom_dim * self.label_size)).cuda())  # 192*96*label_size
            self.b3 = Parameter(torch.randn((self.zoom_dim * self.label_size)).cuda())
            self.b_hyper2 = Parameter(torch.randn(self.z_dim, self.label_size)).cuda()

        if self.net_type == 'onebias':
            self.b_hyper3 = Parameter(torch.randn(self.z_dim, self.label_size)).cuda()

        if self.net_type == 'twobias':
            self.w4 = Parameter(torch.randn((self.z_dim, self.hidden)).cuda())  # 192*96*label_size
            self.b4 = Parameter(torch.randn((self.hidden)).cuda())

            self.b_hyper4 = Parameter(torch.randn(self.hidden, self.label_size)).cuda()

    def forward(self, z):

        # print("Z size",z.shape)
        if self.net_type == 'twoM+bias':
            h_in = torch.matmul(z, self.w2) + self.b2
            h_in = h_in.view(-1, self.hidden)
            h_in = self.bn_h_in(h_in)

            h_final = torch.matmul(h_in, self.w1) + self.b1
            h_final = self.bn_kernel(h_final)
            kernel = h_final.view(-1, self.zoom_dim, self.label_size)

            bias = torch.matmul(h_in, self.b_hyper1)
            bias = self.bn_bias(bias)

        if self.net_type == 'oneM+bias':
            h_final = torch.matmul(z, self.w3) + self.b3
            kernel = h_final.view(-1, self.zoom_dim, self.label_size)

            bias = torch.matmul(z, self.b_hyper2)
            # import pdb;pdb.set_trace()

        if self.net_type == 'onebias':
            bias = torch.matmul(z, self.b_hyper3)
            return None, bias

        if self.net_type == 'twobias':
            h_in = torch.matmul(z, self.w4) + self.b4

            bias = torch.matmul(h_in, self.b_hyper4)
            return None, bias

        # print("bias",bias.shape)

        return kernel, bias


class Embedding(nn.Module):

    def __init__(self, input_size=192, output_size=1501, batch_size=32, net_type=default_net_type):
        super(Embedding, self).__init__()
        # self.hyper_linear=nn.Linear(input_size, output_size, bias=False)
        self.batch_size = batch_size
        self.net_type = net_type

    def forward(self, x, para, bias):
        if self.net_type == 'twoM+bias' or self.net_type == 'oneM+bias':
            outputs = (torch.bmm(x.unsqueeze(1),
                                 para).squeeze() + bias)  # 32*1*96;32*96*label_size  + 32*label_size------> 32*1*label_size

        if self.net_type == 'onebias' or self.net_type == 'twobias':
            outputs = x + bias
        # print("outputs",outputs.shape)
        # import pdb;pdb.set_trace()
        return outputs


def ash_b(x, percentile=90):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def validate_one_step(model, clip, labels, flow, model_flow):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip)  # 16,1024,8,14,14
        v_feat = (x_slow.detach(), x_fast.detach())  # slow 16,1280,16,14,14, fast 16,128,64,14,14

        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd, v_predict_masked, v_emb_masked = model.module.cls_head(v_feat)

        f_feat = model_flow.module.backbone.get_feature(flow)  # 16,1024,8,14,14
        f_feat = model_flow.module.backbone.get_predict(f_feat)
        f_predict, f_emd = model_flow.module.cls_head(f_feat)

        if args.use_ash:
            v_emd = ash_b(v_emd.view(v_emd.size(0), -1, 1, 1))
            v_emd = v_emd.view(v_emd.size(0), -1)
            f_emd = ash_b(f_emd.view(f_emd.size(0), -1, 1, 1))
            f_emd = f_emd.view(f_emd.size(0), -1)
            v_emb_masked = ash_b(v_emb_masked.view(v_emb_masked.size(0), -1, 1, 1))
            v_emb_masked = v_emb_masked.view(v_emb_masked.size(0), -1)

        if args.use_react:
            v_emd = v_emd.clip(max=args.v_thr)
            v_emd = v_emd.view(v_emd.size(0), -1)
            f_emd = f_emd.clip(max=args.f_thr)
            f_emd = f_emd.view(f_emd.size(0), -1)
            v_emb_masked = v_emb_masked.clip(max=args.v_thr)
            v_emb_masked = v_emb_masked.view(v_emb_masked.size(0), -1)

        if True:
            agg_embedding = torch.cat((v_emd, f_emd), dim=1)
            agg_embedding_masked = torch.cat((v_emb_masked, f_emd), dim=1)
            para, bias = hyper_network(agg_embedding_masked)
            predict = embedding(agg_embedding, para, bias)

        feature = torch.cat((v_emd, f_emd), dim=1)

    return predict, feature, v_predict, f_predict


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)

    def forward(self, vfeat, afeat):
        feat = torch.cat((vfeat, afeat), dim=1)
        return self.enc_net(feat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='/data/shared_dataset/data/EPIC_KITCHENS/',
                        help='datapath')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--resumef", type=str, default='checkpoint.pt')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument('--use_ash', action='store_true')
    parser.add_argument('--use_react', action='store_true')
    parser.add_argument('--v_thr', type=float, default=0.5246642529964447,
                        help='v_thr')
    parser.add_argument('--f_thr', type=float, default=0.5100213885307313,
                        help='f_thr')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default='HMDB')  # HMDB Kinetics
    parser.add_argument('--far_ood', action='store_true')
    parser.add_argument("--ood_dataset", type=str, default='EPIC')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.use_react:
        percentile = 90
        if args.far_ood:
            feature_name = '../HMDB-rgb-flow/saved_files/id_' + args.dataset + '_feature_' + args.appen + 'val.npy'
        else:
            feature_name = 'saved_files/id_' + args.ood_dataset + '_near_ood_feature_' + args.appen + 'val.npy'
        id_train_feature = np.load(feature_name)
        v_emd = id_train_feature[:, :2304]
        args.v_thr = np.percentile(v_emd.flatten(), percentile)
        f_emd = id_train_feature[:, 2304:]
        args.f_thr = np.percentile(f_emd.flatten(), percentile)

        args.appen = args.appen + 'react_'

    if args.use_ash:
        args.appen = args.appen + 'ash_'

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'

    device = 'cuda:0'  # or 'cpu'
    device = torch.device(device)

    v_dim = 2304
    f_dim = 2048

    if args.far_ood:
        if args.dataset == 'HMDB':
            num_class = 43
        elif args.dataset == 'Kinetics':
            num_class = 229
    else:
        num_class = 4

    # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)
    type = default_net_type

    if args.dataset == 'Kinetics':
        zoom_size = v_dim + f_dim
        type = 'twoM+bias'
        # import pdb;pdb.set_trace()
        if not args.far_ood:
            hidden_size = 3584
        else:
            hidden_size = 1856
    else:
        zoom_size = v_dim + f_dim
    print('Args.far_ood:' + str(args.far_ood))
    hyper_network = HyperNetwork(z_dim=v_dim + f_dim, label_size=num_class, net_type=type,
                                 zoom_dim=zoom_size)
    embedding = Embedding(input_size=v_dim + f_dim, output_size=num_class, net_type=type)

    model_flow = init_recognizer(config_file_flow, device=device, use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
    cfg_flow = model_flow.cfg
    model_flow = torch.nn.DataParallel(model_flow)
    '''
    mlp_cls = Encoder(input_dim=v_dim+f_dim, out_dim=num_class)
    mlp_cls = mlp_cls.cuda()
    '''
    resume_file = args.resumef
    batch_size = args.bsz
    print("Resuming from ", resume_file)
    checkpoint = torch.load(resume_file)
    BestTestAcc = checkpoint['BestTestAcc']

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model_flow.load_state_dict(checkpoint['model_flow_state_dict'], strict=False)
    if args.appen != 'Rand':
        print('载入了')
        hyper_network.load_state_dict(checkpoint['hyper_network_state_dict'])
    # mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])

    model.eval()
    model_flow.eval()
    hyper_network.eval()
    # mlp_cls.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.far_ood:
        eval_dataset = EPICDOMAIN(split='eval', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath,
                                  far_ood=args.far_ood)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=(device.type == "cuda"), drop_last=False)
        dataloaders = {'eval': eval_dataloader}
        splits = ['eval']
    else:
        train_dataset = EPICDOMAIN(split='train', eval=True, cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       num_workers=args.num_workers, shuffle=False,
                                                       pin_memory=(device.type == "cuda"), drop_last=False)

        val_dataset = EPICDOMAIN(split='val', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers,
                                                     shuffle=False,
                                                     pin_memory=(device.type == "cuda"), drop_last=False)
        test_dataset = EPICDOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=(device.type == "cuda"), drop_last=False)

        eval_dataset = EPICDOMAIN(split='eval', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=(device.type == "cuda"), drop_last=False)
        dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader,
                       'eval': eval_dataloader}
        splits = ['test', 'eval', 'train', 'val']

    for split in splits:
        print(split)
        pred_list, conf_list, label_list, output_list, feature_list = [], [], [], [], []
        for clip, spectrogram, labels in tqdm(dataloaders[split]):
            output, feature, output_v, output_f = validate_one_step(model, clip, labels, spectrogram, model_flow)
            score = torch.softmax(output, dim=1)
            conf, pred = torch.max(score, dim=1)
            output_list.append(output.cpu())
            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(labels.cpu())
            feature_list.append(feature.cpu())

        output_list = torch.cat(output_list).numpy()
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        feature_list = torch.cat(feature_list).numpy()

        if args.far_ood:
            output_name = 'saved_files/id_' + args.dataset + '_ood_' + args.ood_dataset + '_output_' + args.appen + split + '.npy'
            pred_name = 'saved_files/id_' + args.dataset + '_ood_' + args.ood_dataset + '_pred_' + args.appen + split + '.npy'
            conf_name = 'saved_files/id_' + args.dataset + '_ood_' + args.ood_dataset + '_conf_' + args.appen + split + '.npy'
            label_name = 'saved_files/id_' + args.dataset + '_ood_' + args.ood_dataset + '_label_' + args.appen + split + '.npy'
            feature_name = 'saved_files/id_' + args.dataset + '_ood_' + args.ood_dataset + '_feature_' + args.appen + split + '.npy'
        else:
            output_name = 'saved_files/id_' + args.ood_dataset + '_near_ood_output_' + args.appen + split + '.npy'
            pred_name = 'saved_files/id_' + args.ood_dataset + '_near_ood_pred_' + args.appen + split + '.npy'
            conf_name = 'saved_files/id_' + args.ood_dataset + '_near_ood_conf_' + args.appen + split + '.npy'
            label_name = 'saved_files/id_' + args.ood_dataset + '_near_ood_label_' + args.appen + split + '.npy'
            feature_name = 'saved_files/id_' + args.ood_dataset + '_near_ood_feature_' + args.appen + split + '.npy'

        np.save(output_name, output_list)
        np.save(pred_name, pred_list)
        np.save(conf_name, conf_list)
        np.save(label_name, label_list)
        np.save(feature_name, feature_list)

