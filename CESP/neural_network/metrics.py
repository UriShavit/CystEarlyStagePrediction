from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=0.00001):
    """
    Standard Dice coefficient.
    Used to gauge the similarity of two samples.
    Read more in: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
           (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)


def dice_soft(y_true, y_pred, smooth=0.00001):
    """
    Soft Dice coefficient
    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    # Identify axis
    axis = identify_axis(y_true.get_shape())

    # Calculate required variables
    intersection = y_true * y_pred
    intersection = K.sum(intersection, axis=axis)
    y_true = K.sum(y_true, axis=axis)
    y_pred = K.sum(y_pred, axis=axis)

    # Calculate Soft Dice Similarity Coefficient
    dice = ((2 * intersection) + smooth) / (y_true + y_pred + smooth)

    # Obtain mean of Dice & return result score
    dice = K.mean(dice)
    return dice


def dice_soft_loss(y_true, y_pred):
    return 1-dice_soft(y_true, y_pred)


def dice_weighted(weights):
    weights = K.variable(weights)

    def weighted_loss(y_true, y_pred, smooth=0.00001):
        axis = identify_axis(y_true.get_shape())
        intersection = y_true * y_pred
        intersection = K.sum(intersection, axis=axis)
        y_true = K.sum(y_true, axis=axis)
        y_pred = K.sum(y_pred, axis=axis)
        dice = ((2 * intersection) + smooth) / (y_true + y_pred + smooth)
        dice = dice * weights
        return -dice
    return weighted_loss


def dice_crossentropy(y_truth, y_pred):
    # Obtain Soft DSC
    dice = dice_soft_loss(y_truth, y_pred)
    # Obtain Crossentropy
    crossentropy = K.categorical_crossentropy(y_truth, y_pred)
    crossentropy = K.mean(crossentropy)
    # Return sum
    return dice + crossentropy

#-----------------------------------------------------#
#                    Tversky loss                     #
#-----------------------------------------------------#
#                     Reference:                      #
#                Sadegh et al. (2017)                 #
#     Tversky loss function for image segmentation    #
#      using 3D fully convolutional deep networks     #
#-----------------------------------------------------#
# alpha=beta=0.5 : dice coefficient                   #
# alpha=beta=1   : jaccard                            #
# alpha+beta=1   : produces set of F*-scores          #
#-----------------------------------------------------#
def tversky_loss(y_true, y_pred, smooth=0.000001):
    """
    Tversky loss function for image segmentation using 3D fully convolutional deep networks
    Read more: https://arxiv.org/abs/1706.05721
    alpha=beta=0.5 : dice coefficient  - What we are using
    alpha=beta=1   : jaccard
    alpha+beta=1   : produces set of F*-scores
    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    # Define alpha and beta
    alpha = 0.5
    beta  = 0.5
    # Calculate Tversky for each class
    axis = identify_axis(y_true.get_shape())
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth)/(tp + alpha*fn + beta*fp + smooth)
    # Sum up classes to one score
    tversky = K.sum(tversky_class, axis=[-1])
    # Identify number of classes
    n = K.cast(K.shape(y_true)[-1], 'float32')
    # Return Tversky
    return n-tversky


def tversky_crossentropy(y_truth, y_pred):
    # Obtain Tversky Loss
    tversky = tversky_loss(y_truth, y_pred)
    # Obtain Crossentropy
    crossentropy = K.categorical_crossentropy(y_truth, y_pred)
    crossentropy = K.mean(crossentropy)
    # Return sum
    return tversky + crossentropy


def identify_axis(shape):
    """
    Subroutine to identify shape of tensor and return correct axes
    :param shape:
    :return:
    """
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
