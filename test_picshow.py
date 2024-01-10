from torchvision import transforms
import os


def show(x, batch_idx, sup_name, net_name):
    predict = (x > 0.5)
    outpic = predict + 0.0
    im = outpic.squeeze(0)
    trans_pil = transforms.ToPILImage()
    image_pil = trans_pil(im)
    model_dir = os.path.join(os.getcwd(), 'feature_map', str(net_name), str(sup_name), str(batch_idx) + '.jpg')
    image_pil.save(model_dir)


def picshow7(out, s1, s2, s3, s4, s5, s6, batch_idx, net_name):
    show(out, batch_idx, 'sup0', net_name)
    show(s1, batch_idx, 'sup1', net_name)
    show(s2, batch_idx, 'sup2', net_name)
    show(s3, batch_idx, 'sup3', net_name)
    show(s4, batch_idx, 'sup4', net_name)
    show(s5, batch_idx, 'sup5', net_name)
    show(s6, batch_idx, 'sup6', net_name)


def picshow1(out, batch_idx, net_name):
    show(out, batch_idx, 'out', net_name)
