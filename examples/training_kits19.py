from CESP.data_loading.interfaces.nifti_io import NIFTI_interface
from CESP.data_loading.data_io import Data_IO
from CESP.processing.data_augmentation import Data_Augmentation
from CESP.processing.subfunctions.normalization import Normalization
from CESP.processing.subfunctions.clipping import Clipping
from CESP.processing.subfunctions.resampling import Resampling
from CESP.processing.preprocessor import Preprocessor
from CESP.neural_network.model import Neural_Network
from CESP.neural_network.metrics import dice_soft, dice_crossentropy, tversky_loss
from tensorflow.keras.callbacks import ReduceLROnPlateau
# Initialize the NIfTI I/O interface and configure the images as one channel (grayscale) and three segmentation classes (background, kidney, tumor)
interface = NIFTI_interface(pattern="case_00[0-9]*",
                            channels=1, classes=3)

data_path = "../../kits19_no_gz/data"
# Create the Data I/O object
data_io = Data_IO(interface, data_path)

sample_list = data_io.get_indiceslist()
sample_list.sort()
print('All samples: {}'.format(str(sample_list)))

# Library import


# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=2, scaling=True, rotations=True, elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True, gaussian_noise=True)


print("Initiating data augmentations...")
# Create a pixel value normalization Subfunction through Z-Score
sf_normalize = Normalization()
# Create a clipping Subfunction between -79 and 304
sf_clipping = Clipping(min=-79, max=304)
# Create a resampling Subfunction to voxel spacing 3.22 x 1.62 x 1.62
sf_resample = Resampling((3.22, 1.62, 1.62))

subfunctions = [sf_resample, sf_clipping, sf_normalize]

print("Initiating Preprocessor...")



# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=2, subfunctions=subfunctions, prepare_subfunctions=True,
                  prepare_batches=False, analysis="patchwise-crop", patch_shape=(80, 160, 160))

# Adjust the patch overlap for predictions
pp.patchwise_overlap = (40, 80, 80)

print("Initiating Neural Network model...")



# Create the Neural Network model
model = Neural_Network(preprocessor=pp, loss=tversky_loss, metrics=[dice_soft, dice_crossentropy],
                       batch_queue_size=3, workers=3, learning_rate=0.0001)

print("Initiating Callbacks...")

cb_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=1, mode='min', min_delta=0.0001, cooldown=1,
                          min_lr=0.00001)
del sample_list[65]
del sample_list[101]
del sample_list[42]
del sample_list[35]
del sample_list[60]
del sample_list[74]
del sample_list[27]
del sample_list[77]

# Create the validation sample ID list
validation_samples = sample_list[0:120]
# Output validation samples
print("Validation samples: " + str(validation_samples))


# Library import
from CESP.evaluation.cross_validation import cross_validation
# Run cross-validation function
cross_validation(validation_samples, model, k_fold=3, epochs=350, iterations=150,
                 evaluation_path="evaluation", draw_figures=True, callbacks=[cb_lr])

from IPython.display import Image
Image(filename = "evaluation/fold_0/validation.dice_soft.png")