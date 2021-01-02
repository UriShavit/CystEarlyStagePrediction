from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from CESP.neural_network.metrics import dice_soft, tversky_loss
from CESP.neural_network.architecture.unet.standard import Architecture
from CESP.neural_network.data_generator import DataGenerator


# Class which represents the Neural Network and which run the whole pipeline
class Neural_Network:
    def __init__(self, preprocessor, architecture=Architecture(), loss=tversky_loss, metrics=[dice_soft],
                 learning_rate=0.0001, batch_queue_size=2, workers=1, gpu_number=1):
        """

        :param preprocessor: Preprocessor class instance which provides the Neural Network with batches.
        :param architecture: Instance of a neural network model Architecture class instance.
                                                By default, a standard U-Net is used as Architecture.
        :param loss: The metric function which is used as loss for training.
                        Any Metric Function defined in Keras, in CESP.neural_network.metrics or any custom
                        metric function, which follows the Keras metric guidelines, can be used.
        :param metrics: List of one or multiple Metric Functions, which will be shown during training.
                        Any Metric Function defined in Keras, in CESP.neural_network.metrics or any custom
                        metric function, which follows the Keras metric guidelines, can be used.
        :param learning_rate: Learning rate in which weights of the neural network will be updated.
        :param batch_queue_size: The batch queue size is the number of previously prepared batches in the cache during runtime.
        :param workers: Number of workers/threads which preprocess batches during runtime.
        :param gpu_number: Number of GPUs, which will be used for training.
        """
        # Identify data parameters
        self.three_dim = preprocessor.data_io.interface.three_dim
        self.channels = preprocessor.data_io.interface.channels
        self.classes = preprocessor.data_io.interface.classes
        # Assemble the input shape
        input_shape = (None,)
        # Initialize model for 3D data
        if self.three_dim:
            input_shape = (None, None, None, self.channels)
            self.model = architecture.create_model_3D(input_shape=input_shape,
                                                      n_labels=self.classes)
         # Initialize model for 2D data
        else:
             input_shape = (None, None, self.channels)
             self.model = architecture.create_model_2D(input_shape=input_shape,
                                                       n_labels=self.classes)
        # Transform to Keras multi GPU model
        if gpu_number > 1:
            self.model = multi_gpu_model(self.model, gpu_number)
        # Compile model
        self.model.compile(optimizer=Adam(lr=learning_rate),
                           loss=loss, metrics=metrics)
        # Cache starting weights
        self.initialization_weights = self.model.get_weights()
        # Cache parameter
        self.preprocessor = preprocessor
        self.loss = loss
        self.metrics = metrics
        self.learninig_rate = learning_rate
        self.batch_queue_size = batch_queue_size
        self.workers = workers

    #---------------------------------------------#
    #               Class variables               #
    #---------------------------------------------#
    shuffle_batches = True                  # Option whether batch order should be shuffled or not
    initialization_weights = None           # Neural Network model weights for weight reinitialization


    def train(self, sample_list, epochs=20, iterations=None, callbacks=[]):
        """
        Fitting function for the Neural Network model using the provided list of sample indices.
        :param sample_list: A list of sample indicies which will be used for training
        :param epochs: Number of epochs. A single epoch is defined as one iteration through the complete data set.
        :param iterations: Number of iterations (batches) in a single epoch.
        :param callbacks: A list of Callback classes for custom evaluation
        :return:
        """
        # Initialize Keras Data Generator for generating batches
        dataGen = DataGenerator(sample_list, self.preprocessor, training=True,
                                validation=False, shuffle=self.shuffle_batches,
                                iterations=iterations)
        # Run training process with Keras fit
        self.model.fit(dataGen,
                       epochs=epochs,
                       callbacks=callbacks,
                       workers=self.workers,
                       max_queue_size=self.batch_queue_size)
        # Clean up temporary files if necessary
        if self.preprocessor.prepare_batches or self.preprocessor.prepare_subfunctions:
            self.preprocessor.data_io.batch_cleanup()


    def predict(self, sample_list, return_output=False, activation_output=False):
        """
        Prediction function for the Neural Network model. The fitted model will predict a segmentation
        for the provided list of sample indices.
        :param sample_list: A list of sample indicies for which a segmentation prediction will be computed
        :param return_output: Parameter which decides, if computed predictions will be output as the return of this
                                function or if the predictions will be saved with the save_prediction method defined
                                in the provided Data I/O interface.
        :param activation_output: Parameter which decides, if model output (activation function, normally softmax) will
                                be saved/outputed (if FALSE) or if the resulting class label (argmax) should be outputed.
        :return:
        """
        # Initialize result array for direct output
        if return_output : results = []
        # Iterate over each sample
        for sample in sample_list:
            # Initialize Keras Data Generator for generating batches
            dataGen = DataGenerator([sample], self.preprocessor,
                                    training=False, validation=False,
                                    shuffle=False, iterations=None)
            # Run prediction process with Keras predict
            pred_list = []
            for batch in dataGen:
                pred_batch = self.model.predict_on_batch(batch)
                pred_list.append(pred_batch)
            pred_seg = np.concatenate(pred_list, axis=0)
            # Postprocess prediction
            pred_seg = self.preprocessor.postprocessing(sample, pred_seg,
                                                        activation_output)
            # Backup predicted segmentation
            if return_output : results.append(pred_seg)
            else : self.preprocessor.data_io.save_prediction(pred_seg, sample)
            # Clean up temporary files if necessary
            if self.preprocessor.prepare_batches or self.preprocessor.prepare_subfunctions:
                self.preprocessor.data_io.batch_cleanup()
        # Output predictions results if direct output modus is active
        if return_output : return results


    def evaluate(self, training_samples, validation_samples, epochs=20, iterations=None, callbacks=[]):
        """
        Evaluate the Neural Network model using the CESP pipeline
        valuation function for the Neural Network model using the provided lists of sample indices
        for training and validation. It is also possible to pass custom Callback classes in order to
        obtain more information.

        :param training_samples: A list of sample indicies which will be used for training
        :param validation_samples: A list of sample indicies which will be used for validation
        :param epochs: Number of epochs. A single epoch is defined as one iteration through the complete data set.
        :param iterations: Number of iterations (batches) in a single epoch.
        :param callbacks: A list of Callback classes for custom evaluation
        :return:  Gathered fitting information and evaluation results of the validation
        """
        # Initialize a Keras Data Generator for generating Training data
        dataGen_training = DataGenerator(training_samples, self.preprocessor,
                                         training=True, validation=False,
                                         shuffle=self.shuffle_batches,
                                         iterations=iterations)
        # Initialize a Keras Data Generator for generating Validation data
        dataGen_validation = DataGenerator(validation_samples,
                                           self.preprocessor,
                                           training=True, validation=True,
                                           shuffle=self.shuffle_batches)
        # Run training & validation process with the Keras fit
        history = self.model.fit(dataGen_training,
                                 validation_data=dataGen_validation,
                                 callbacks=callbacks,
                                 epochs=epochs,
                                 workers=self.workers,
                                 max_queue_size=self.batch_queue_size)
        # Clean up temporary files if necessary
        if self.preprocessor.prepare_batches or self.preprocessor.prepare_subfunctions:
            self.preprocessor.data_io.batch_cleanup()
        # Return the training & validation history
        return history


    def reset_weights(self):
        self.model.set_weights(self.initialization_weights)

    # Dump model to file
    def dump(self, file_path):
        self.model.save(file_path)

    # Load model from file
    def load(self, file_path, custom_objects={}):
        # Create model input path
        self.model = load_model(file_path, custom_objects, compile=False)
        # Compile model
        self.model.compile(optimizer=Adam(lr=self.learninig_rate),
                           loss=self.loss, metrics=self.metrics)
