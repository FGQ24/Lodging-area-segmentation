# # # 别动啦！！！
import os
import numpy as np
import torch
# import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_preprocess import data_augment
from test_picshow import picshow7, picshow1
from leaf_seg import pic_seg
# 网络选择
from Net_wode import wode
from Net_U2 import U2_net
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
    # 设置GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TO_MODIFY 网络选择
    net = wode().to("cuda:0")
    net_name = 'wode'
    # TO_MODIFY 参数加载
    para_name = 'wode_bce_itr_80'
    model_dir = os.path.join(os.getcwd(), 'saved_models', para_name + '.pth')
    net.load_state_dict(torch.load(model_dir, map_location=device))
    net = net.to("cuda:0")
    # 载入保存的模型参数
    net.eval()
    # 不启用 BatchNormalization 和 Dropout
    # 数据库载入
    test_data = data_augment('cropped_images/3', "image", "label")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    # 获取图像文件名称
    img_list = test_data.img_list

    total_pix = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.2fM" % (total_pix / 1e6))

    # # visdom启动命令：python -m visdom.server
    # # 初始化窗口
    # viz = visdom.Visdom(env='plot_test')
    # # 创建线图初始点。Ps核心准确率图，acc精度图，pre查准率图，rec召回率图，IoU交并比图
    # viz.line([0.], [0.], win='test_Ps', opts=dict(title='test_Ps'))
    # viz.line([0.], [0.], win='test_rec', opts=dict(title='test_rec'))
    # viz.line([0.], [0.], win='test_pre', opts=dict(title='test_pre'))
    # viz.line([0.], [0.], win='test_F1', opts=dict(title='test_F1'))
    # viz.line([0.], [0.], win='test_IoU', opts=dict(title='test_IoU'))
    # global_step = 0.0

    # # # test
    acc = []
    rec = []
    pre = []
    F1 = []
    IoU = []
    Dice = []
    num = 0
    bidx = -1
    total_num = len(img_list)
    for batch_idx, (x, y) in enumerate(test_loader):
        # 获取图像文件名称
        bidx = bidx + 1
        img_name = img_list[bidx]
        img_name = img_name.split('.png')[0]
        # x : [batch, 1, 512, 512] → y:[batch, 1, 512 ,512]
        X = get_variable(x).to(device)
        Y = get_variable(y).to(device)
        # X = X.permute(0, 3, 1, 2)
        # Y = Y.permute(0, 3, 1, 2)
        image = torch.as_tensor(X, dtype=torch.float32)
        label = torch.as_tensor(Y, dtype=torch.float32)

        out = None

        if net_name in ['U_net', 'Seg_net']:
            out = net(image)
            picshow1(out, img_name, net_name)
        elif net_name in ['U2_net', 'ATU2_net', 'U2s_net', 'wode']:
            out, s1, s2, s3, s4, s5, s6 = [t.to(device) for t in net(image)]
            picshow7(out, s1, s2, s3, s4, s5, s6, img_name, net_name)

        # set threshold to 0.5
        threshold = 0.5
        # 预测转0，1
        predict = (out > 0.5)

        ##
        pic_seg(image, out, bidx, net_name)
        ##

        # TP : predict 和 y 均为 1
        TPP = (predict * label).cpu().detach().numpy()
        TP = np.nan_to_num(TPP)
        TP = torch.from_numpy(TP)
        TP = torch.sum(TP == 1)

        # TN : predict 和 label 均为 0
        TNN = (predict + label).cpu().detach().numpy()
        TN = np.nan_to_num(TNN)
        TN = torch.from_numpy(TN)
        TN = torch.sum(TN == 0)

        # FP : predict 为 1，y 为 0
        FPP = torch.sum(predict == 1)
        FP = (FPP - TP)

        # FN : predict 为 0，y 为 1
        FNN = torch.sum(predict == 0)
        FN = (FNN - TN)

        # # # 评价指标
        # Ps ： (TP + TN) / (TP + TN + FP + FN)
        Ps = ((TP + TN) / (TP + TN + FP + FN)).cpu().detach().numpy()
        Ps = np.nan_to_num(Ps)
        acc.append(Ps)

        # # # Recall
        recall = (TP / (TP + FN)).cpu().detach().numpy()
        recall = np.nan_to_num(recall)

        # # # precision

        precision = (TP / (TP + FP)).cpu().detach().numpy()
        precision = np.nan_to_num(precision)

        # # # F1
        F0 = (2 * (TP / (TP + FN)) * (TP / (TP + FP))) / torch.sum((TP / (TP + FN)) + (TP / (TP + FP)))
        F0 = F0.cpu()  # 将张量复制到CPU
        F0 = np.nan_to_num(F0.numpy())

        # # # 类1 IoU
        IoUo = TP / (TP + FN + FP).cpu().detach().numpy()
        IoUo = np.nan_to_num(IoUo)

        # # # Dice
        Dicee = (2 * IoUo) / (1 + IoUo)
        Dicee = np.nan_to_num(Dicee)

        if ((Ps == 1) and (recall != 0)) or (Ps != 1):
            rec.append(recall)

            pre.append(precision)

            F1.append(F0)

            IoU.append(IoUo)

            Dice.append(Dicee)
            num = num + 1

            # global_step += 1
            # viz.line([Ps.item()], [global_step], win='test_Ps', update='append')
            # viz.line([recall.item()], [global_step], win='test_rec', update='append')
            # viz.line([precision.item()], [global_step], win='test_pre', update='append')
            # viz.line([F0.item()], [global_step], win='test_F1', update='append')
            # viz.line([IoUo.item()], [global_step], win='test_IoU', update='append')
        # # # 结果呈现
        print(str(bidx + 1) + ' ' + str(img_name) + ' ' + str(Ps.item()))
    final_acc = torch.sum(torch.tensor(acc)) / total_num
    final_rec = torch.sum(torch.tensor(rec)) / num
    final_pre = torch.sum(torch.tensor(pre)) / num
    final_F1 = torch.sum(torch.tensor(F1)) / num
    final_IoU = torch.sum(torch.tensor(IoU)) / num
    final_Dice = torch.sum(torch.tensor(Dice)) / num

    print('本次测试的准确率为' + str(final_acc))
    print('本次测试的召回率为' + str(final_rec))
    print('本次测试的查准率为' + str(final_pre))
    print('本次测试的F1值为' + str(final_F1))
    print('本次测试的类1IoU值为' + str(final_IoU))
    print('本次测试的Dice值为' + str(final_Dice))
