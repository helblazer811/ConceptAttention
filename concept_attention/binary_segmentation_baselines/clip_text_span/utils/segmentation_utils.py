import torch
import matplotlib.cm
import skimage.io
import skimage.feature
import skimage.filters
import numpy as np
import os
from collections import OrderedDict
import glob
from sklearn.metrics import f1_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve

SMOOTH = 1e-6


def get_iou(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.cpu().numpy()


def get_f1_scores(predict, target, ignore_index=-1):
    # Tensor process
    batch_size = predict.shape[0]
    predict = predict.data.cpu().numpy().reshape(-1)
    target = target.data.cpu().numpy().reshape(-1)
    pb = predict[target != ignore_index].reshape(batch_size, -1)
    tb = target[target != ignore_index].reshape(batch_size, -1)

    total = []
    for p, t in zip(pb, tb):
        total.append(np.nan_to_num(f1_score(t, p)))

    return total


def get_roc(predict, target, ignore_index=-1):
    target_expand = target.unsqueeze(1).expand_as(predict)
    target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)
    # Tensor process
    x = torch.zeros_like(target_expand)
    t = target.unsqueeze(1).clamp(min=0)
    target_1hot = x.scatter_(1, t, 1)
    batch_size = predict.shape[0]
    predict = predict.data.cpu().numpy().reshape(-1)
    target = target_1hot.data.cpu().numpy().reshape(-1)
    pb = predict[target_expand_numpy != ignore_index].reshape(batch_size, -1)
    tb = target[target_expand_numpy != ignore_index].reshape(batch_size, -1)

    total = []
    for p, t in zip(pb, tb):
        total.append(roc_curve(t, p))

    return total


def get_pr(predict, target, ignore_index=-1):
    target_expand = target.unsqueeze(1).expand_as(predict)
    target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)
    # Tensor process
    x = torch.zeros_like(target_expand)
    t = target.unsqueeze(1).clamp(min=0)
    target_1hot = x.scatter_(1, t, 1)
    batch_size = predict.shape[0]
    predict = predict.data.cpu().numpy().reshape(-1)
    target = target_1hot.data.cpu().numpy().reshape(-1)
    pb = predict[target_expand_numpy != ignore_index].reshape(batch_size, -1)
    tb = target[target_expand_numpy != ignore_index].reshape(batch_size, -1)

    total = []
    for p, t in zip(pb, tb):
        total.append(precision_recall_curve(t, p))

    return total


def get_ap_scores(predict, target, ignore_index=-1):
    total = []
    for pred, tgt in zip(predict, target):
        target_expand = tgt.unsqueeze(0).expand_as(pred)
        target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)

        # Tensor process
        x = torch.zeros_like(target_expand)
        t = tgt.unsqueeze(0).clamp(min=0).long()
        target_1hot = x.scatter_(0, t, 1)
        predict_flat = pred.data.cpu().numpy().reshape(-1)
        target_flat = target_1hot.data.cpu().numpy().reshape(-1)

        p = predict_flat[target_expand_numpy != ignore_index]
        t = target_flat[target_expand_numpy != ignore_index]

        total.append(np.nan_to_num(average_precision_score(t, p)))

    return total


def get_ap_multiclass(predict, target):
    total = []
    for pred, tgt in zip(predict, target):
        predict_flat = pred.data.cpu().numpy().reshape(-1)
        target_flat = tgt.data.cpu().numpy().reshape(-1)

        total.append(np.nan_to_num(average_precision_score(target_flat, predict_flat)))

    return total


def batch_precision_recall(predict, target, thr=0.5):
    """Batch Precision Recall
    Args:
        predict: input 4D tensor
        target: label 4D tensor
    """
    # _, predict = torch.max(predict, 1)

    predict = predict > thr
    predict = predict.data.cpu().numpy() + 1
    target = target.data.cpu().numpy() + 1

    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))

    precision = float(np.nan_to_num(tp / (tp + fp)))
    recall = float(np.nan_to_num(tp / (tp + fn)))

    return precision, recall


def batch_pix_accuracy(predict, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 3D tensor
        target: label 3D tensor
    """

    # for thr in np.linspace(0, 1, slices):

    _, predict = torch.max(predict, 0)
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 3D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(predict, 0)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


def pixel_accuracy(im_pred, im_lab):
    # ref https://github.com/CSAILVision/sceneparsing/blob/master/evaluationCode/utils_eval.py
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    # pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled


def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image.
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union:
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.train_dataset, args.model)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['train_dataset'] = self.args.train_dataset
        p['lr'] = self.args.lr
        p['epoch'] = self.args.epochs

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()


class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


def vec2im(V, shape=()):
    '''
    Transform an array V into a specified shape - or if no shape is given assume a square output format.

    Parameters
    ----------

    V : numpy.ndarray
        an array either representing a matrix or vector to be reshaped into an two-dimensional image

    shape : tuple or list
        optional. containing the shape information for the output array if not given, the output is assumed to be square

    Returns
    -------

    W : numpy.ndarray
        with W.shape = shape or W.shape = [np.sqrt(V.size)]*2

    '''

    if len(shape) < 2:
        shape = [np.sqrt(V.size)] * 2
        shape = map(int, shape)
    return np.reshape(V, shape)


def enlarge_image(img, scaling=3):
    '''
    Enlarges a given input matrix by replicating each pixel value scaling times in horizontal and vertical direction.

    Parameters
    ----------

    img : numpy.ndarray
        array of shape [H x W] OR [H x W x D]

    scaling : int
        positive integer value > 0

    Returns
    -------

    out : numpy.ndarray
        two-dimensional array of shape [scaling*H x scaling*W]
        OR
        three-dimensional array of shape [scaling*H x scaling*W x D]
        depending on the dimensionality of the input
    '''

    if scaling < 1 or not isinstance(scaling, int):
        print('scaling factor needs to be an int >= 1')

    if len(img.shape) == 2:
        H, W = img.shape

        out = np.zeros((scaling * H, scaling * W))
        for h in range(H):
            fh = scaling * h
            for w in range(W):
                fw = scaling * w
                out[fh:fh + scaling, fw:fw + scaling] = img[h, w]

    elif len(img.shape) == 3:
        H, W, D = img.shape

        out = np.zeros((scaling * H, scaling * W, D))
        for h in range(H):
            fh = scaling * h
            for w in range(W):
                fw = scaling * w
                out[fh:fh + scaling, fw:fw + scaling, :] = img[h, w, :]

    return out


def repaint_corner_pixels(rgbimg, scaling=3):
    '''
    DEPRECATED/OBSOLETE.

    Recolors the top left and bottom right pixel (groups) with the average rgb value of its three neighboring pixel (groups).
    The recoloring visually masks the opposing pixel values which are a product of stabilizing the scaling.
    Assumes those image ares will pretty much never show evidence.

    Parameters
    ----------

    rgbimg : numpy.ndarray
        array of shape [H x W x 3]

    scaling : int
        positive integer value > 0

    Returns
    -------

    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3]
    '''

    # top left corner.
    rgbimg[0:scaling, 0:scaling, :] = (rgbimg[0, scaling, :] + rgbimg[scaling, 0, :] + rgbimg[scaling, scaling,
                                                                                       :]) / 3.0
    # bottom right corner
    rgbimg[-scaling:, -scaling:, :] = (rgbimg[-1, -1 - scaling, :] + rgbimg[-1 - scaling, -1, :] + rgbimg[-1 - scaling,
                                                                                                   -1 - scaling,
                                                                                                   :]) / 3.0
    return rgbimg


def digit_to_rgb(X, scaling=3, shape=(), cmap='binary'):
    '''
    Takes as input an intensity array and produces a rgb image due to some color map

    Parameters
    ----------

    X : numpy.ndarray
        intensity matrix as array of shape [M x N]

    scaling : int
        optional. positive integer value > 0

    shape: tuple or list of its , length = 2
        optional. if not given, X is reshaped to be square.

    cmap : str
        name of color map of choice. default is 'binary'

    Returns
    -------

    image : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    # create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))

    image = enlarge_image(vec2im(X, shape), scaling)  # enlarge
    image = cmap(image.flatten())[..., 0:3].reshape([image.shape[0], image.shape[1], 3])  # colorize, reshape

    return image


def hm_to_rgb(R, X=None, scaling=3, shape=(), sigma=2, cmap='bwr', normalize=True):
    '''
    Takes as input an intensity array and produces a rgb image for the represented heatmap.
    optionally draws the outline of another input on top of it.

    Parameters
    ----------

    R : numpy.ndarray
        the heatmap to be visualized, shaped [M x N]

    X : numpy.ndarray
        optional. some input, usually the data point for which the heatmap R is for, which shall serve
        as a template for a black outline to be drawn on top of the image
        shaped [M x N]

    scaling: int
        factor, on how to enlarge the heatmap (to control resolution and as a inverse way to control outline thickness)
        after reshaping it using shape.

    shape: tuple or list, length = 2
        optional. if not given, X is reshaped to be square.

    sigma : double
        optional. sigma-parameter for the canny algorithm used for edge detection. the found edges are drawn as outlines.

    cmap : str
        optional. color map of choice

    normalize : bool
        optional. whether to normalize the heatmap to [-1 1] prior to colorization or not.

    Returns
    -------

    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    # create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))

    if normalize:
        R = R / np.max(np.abs(R))  # normalize to [-1,1] wrt to max relevance magnitude
        R = (R + 1.) / 2.  # shift/normalize to [0,1] for color mapping

    R = enlarge_image(R, scaling)
    rgb = cmap(R.flatten())[..., 0:3].reshape([R.shape[0], R.shape[1], 3])
    # rgb = repaint_corner_pixels(rgb, scaling) #obsolete due to directly calling the color map with [0,1]-normalized inputs

    if not X is None:  # compute the outline of the input
        # X = enlarge_image(vec2im(X,shape), scaling)
        xdims = X.shape
        Rdims = R.shape

    return rgb


def save_image(rgb_images, path, gap=2):
    '''
    Takes as input a list of rgb images, places them next to each other with a gap and writes out the result.

    Parameters
    ----------

    rgb_images : list , tuple, collection. such stuff
        each item in the collection is expected to be an rgb image of dimensions [H x _ x 3]
        where the width is variable

    path : str
        the output path of the assembled image

    gap : int
        optional. sets the width of a black area of pixels realized as an image shaped [H x gap x 3] in between the input images

    Returns
    -------

    image : numpy.ndarray
        the assembled image as written out to path
    '''

    sz = []
    image = []
    for i in range(len(rgb_images)):
        if not sz:
            sz = rgb_images[i].shape
            image = rgb_images[i]
            gap = np.zeros((sz[0], gap, sz[2]))
            continue
        if not sz[0] == rgb_images[i].shape[0] and sz[1] == rgb_images[i].shape[2]:
            print('image', i, 'differs in size. unable to perform horizontal alignment')
            print('expected: Hx_xD = {0}x_x{1}'.format(sz[0], sz[1]))
            print('got     : Hx_xD = {0}x_x{1}'.format(rgb_images[i].shape[0], rgb_images[i].shape[1]))
            print('skipping image\n')
        else:
            image = np.hstack((image, gap, rgb_images[i]))

    image *= 255
    image = image.astype(np.uint8)

    print('saving image to ', path)
    skimage.io.imsave(path, image)
    return image


class IoU(Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.

        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou)
    