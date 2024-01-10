from torchvision import transforms
import os


def show(original, x, batch_idx, net_name):
    predict = (x > 0.5)
    outpic = predict + 0.0
    im = outpic * original
    img = im.squeeze(0)
    trans_pil = transforms.ToPILImage()
    image_pil = trans_pil(img)
    model_dir = os.path.join(os.getcwd(), 'seged_pic', str(net_name), str(batch_idx) + '.jpg')
    image_pil.save(model_dir)


def pic_seg(original, out, batch_idx, net_name):
    show(original, out, batch_idx, net_name)