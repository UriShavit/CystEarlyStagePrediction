
import numpy

# Object containing an image and the associated segmentation
class Sample:
    # Initialize class variable
    index = None
    img_data = None
    seg_data = None
    pred_data = None
    shape = None
    channels = None
    classes = None
    details = None

    # Create a Sample object
    def __init__(self, index, image, channels, classes):
        # Preprocess image data if required
        if image.shape[-1] != channels:
            image = numpy.reshape(image, image.shape + (channels,))
        # Cache data
        self.index = index
        self.img_data = image
        self.channels = channels
        self.classes = classes
        self.shape = self.img_data.shape

    # Add and preprocess a segmentation annotation
    def add_segmentation(self, seg):
        if seg.shape[-1] != 1:
            seg = numpy.reshape(seg, seg.shape + (1,))
        self.seg_data = seg

    # Add and preprocess a prediction annotation
    def add_prediction(self, pred):
        if pred.shape[-1] != 1:
            pred = numpy.reshape(pred, pred.shape + (1,))
        self.pred_data = pred

    # Add optional information / details for custom usage
    def add_details(self, details):
        self.details = details
