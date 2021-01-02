
import numpy as np
from CESP.data_loading.data_io import create_directories
from CESP.evaluation.detailed_validation import detailed_validation


def leave_one_out(sample_list, model, epochs=20, iterations=None, callbacks=[], evaluation_path="evaluation"):
    """
    Function for an automatic Leave-One-Out Validation of the Neural Network model by
    running the whole pipeline once by training on the complete data set except one sample
    and then predict the segmentation of the last remaining sample.
    :param sample_list: A list of sample indicies which will be used for validation.
    :param model: Instance of a Neural Network model class instance.
    :param epochs: Number of epochs. A single epoch is defined as one iteration through the complete data set.
    :param iterations: Number of iterations (batches) in a single epoch.
    :param callbacks: A list of Callback classes for custom evaluation.
    :param evaluation_path: Path to the evaluation data directory. This directory will be created and used for storing
                            all kinds of evaluation results during the validation processes.
    :return:
    """
    # Choose a random sample
    loo = sample_list.pop(np.random.choice(len(sample_list)))
    # Reset Neural Network model weights
    model.reset_weights()
    # Train the model with the remaining samples
    model.train(sample_list, epochs=epochs, iterations=iterations,
                callbacks=callbacks)
    # Initialize evaluation directory
    create_directories(evaluation_path)
    # Make a detailed validation on the LOO sample
    detailed_validation([loo], model, evaluation_path)
