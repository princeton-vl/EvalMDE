import cv2

from evalmde.utils.common import pathlib_file


def imread_rgb(img_f):
    return cv2.imread(str(pathlib_file(img_f)))[..., ::-1].copy()


def imwrite_rgb(img_f, img, verbose=False):
    img_f = pathlib_file(img_f)
    img_f.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_f), img[..., ::-1])
    if verbose:
        print(f'Saved to {img_f.resolve()}')


def resize(img, H=None, W=None, interpolation=cv2.INTER_NEAREST, return_sc=False):
    '''
    if both H and W are specified, resize to smaller one while keeping aspect ratio
    :param img:
    :param H:
    :param W:
    :param interpolation:
    :param return_sc:
    :return:
    '''
    cur_H, cur_W = img.shape[:2]
    if (H is not None) and (W is not None):
        H = int(H)
        W = int(W)
        if H / cur_H < W / cur_W:
            W = None
        else:
            H = None
    if H is not None:
        H = int(H)
        img = cv2.resize(img, (int(img.shape[1] / img.shape[0] * H), H), interpolation=interpolation)
    if W is not None:
        W = int(W)
        img = cv2.resize(img, (W, int(img.shape[0] / img.shape[1] * W)), interpolation=interpolation)
    if return_sc:
        sc = img.shape[0] / cur_H
        return img, sc
    return img
