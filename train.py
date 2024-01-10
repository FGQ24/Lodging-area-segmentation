# # # 别动啦！！！
import os
import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_preprocess import data_augment
from leaf_seg import pic_seg
from losss import mutil_bce_loss_fusion, mutil_GFocal_loss_fusion,mutil_soft_dice_loss_fusion
from test_picshow import picshow7, picshow1
from utils import init_nor
import visdom
# # 可选网络
from Net_ATU2 import ATU2_net
from Net_U2 import U2_net
from Net_wode import wode
from Net_U import U_net
from Net_Seg import Seg_net
from Net_U2s import U2s_net


def get_variable(variable):
    variable = variable.float()  # ensure variable is a floating poiTN tensor
    variable = Variable(variable, requires_grad=True)
    if torch.cuda.is_available():
        variable = variable.cuda()
    return variable


if __name__ == '__main__':
    # 清除缓存
    torch.cuda.empty_cache()
    # device设定
    device = torch.device("cuda:0")
    # TO_MODIFY 网络选择 ('wode', 'U_net', 'Seg_net', 'U2_net', 'ATU2_net', 'U2s_net')
    net = wode().to(device)
    net_name = 'wode'
    # 网络层参数初始化
    net.apply(init_nor)
    # TO_MODIFY 网络参数内存大小
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    # 优化器Adam
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)

    # # 打印网络模型中的每个参数及其所在的设备
    # for name, param in net.named_parameters():
    #     device = param.device
    #     print(f"Parameter '{name}' is on device: {device}")

    # visdom启动命令：python -m visdom.server
    # # visdom 界面选定
    # viz = visdom.Visdom(env='plot_train')
    # # 创建线图初始点。差值图loss
    # viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))

    global_step = 0.0
    ite_num = 0
    save_frq = 80

    # epoch = 80
    for epoch in range(80):
        net.train()
        ## —————————————————————— ##
        train_data = data_augment("data2/train_data", "image", "label")
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        for batch_idx, (x, y) in enumerate(train_loader):

            X = get_variable(x).to(device)  # 在同一设备上调用 get_variable
            Y = get_variable(y).to(device)  # 在同一设备上调用 get_variable

            image = torch.as_tensor(X, dtype=torch.float32)
            label = torch.as_tensor(Y, dtype=torch.float32)

            optimizer.zero_grad()

            loss = None
            loss0 = None

            if net_name in ['U_net', 'Seg_net', 'DeepLabv3']:
                out = net(image)
                bce_loss = nn.BCELoss(reduction='mean')
                loss = bce_loss(out, label)
            elif net_name in ['U2_net', 'ATU2_net', 'U2s_net']:
                s0, s1, s2, s3, s4, s5, s6 = [t.to(device) for t in net(image)]
                loss, loss0 = mutil_bce_loss_fusion(s0, s1, s2, s3, s4, s5, s6, label)
            elif net_name in ['wode']:
                s0, s1, s2, s3, s4, s5, s6 = [t.to(device) for t in net(image)]
                loss, loss0 = mutil_soft_dice_loss_fusion(s0, s1, s2, s3, s4, s5, s6, label)

            loss.backward()
            optimizer.step()

            # # visdom 绘图
            # global_step += 1
            # viz.line([loss.item()], [global_step], win='train_loss', update='append')

            if net_name in ['U_net', 'Seg_net']:
                if (batch_idx + 10) % 1000 == 0:
                    print(epoch + 1, loss.item())
            elif net_name in ['U2_net', 'ATU2_net', 'U2s_net', 'wode']:
                if (batch_idx + 10) % 1000 == 0:
                    print(epoch + 1, loss0.item())

            model_dir = os.path.join(os.getcwd(), 'saved_models', net_name)
            if (epoch + 1) % save_frq == 0:
                torch.save(net.state_dict(), model_dir + "_bce_itr_%d.pth" % (epoch + 1))
                net.train()  # resume train
