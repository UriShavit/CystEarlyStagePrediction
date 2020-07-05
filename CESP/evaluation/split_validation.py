import numpy as np
import math
from CESP.data_loading.data_io import create_directories, backup_history
from CESP.utils.plotting import plot_validation
from CESP.evaluation.detailed_validation import detailed_validation


def split_validation(sample_list, model, percentage=0.2, epochs=20, iterations=None, evaluation_path="evaluation",
                     draw_figures=False, run_detailed_evaluation=False, callbacks=[], return_output=False):
    """
    Function for an automatic Percentage-Split Validation of the Neural Network model by
    running the whole pipeline once with a test and train data set.
    :param sample_list: A list of sample indicies which will be used for validation.
    :param model: Instance of a Neural Network model class instance.
    :param epochs: Number of epochs. A single epoch is defined as one iteration through the complete data set.
    :param iterations: Number of iterations (batches) in a single epoch.
    :param callbacks: A list of Callback classes for custom evaluation.
    :param evaluation_path: Path to the evaluation data directory. This directory will be created and used for storing
                            all kinds of evaluation results during the validation processes.
    :param percentage: Split percentage of how big the testing data set should be.
                        By default, the percentage value is 0.2 -> 20% testing and 80% training
    :param draw_figures: Option if evaluation figures should be automatically plotted in the evaluation directory.
    :param run_detailed_evaluation: Option if a detailed evaluation (additional prediction) should be performed.
    :param return_output: Option, if computed evaluations will be output as the return of this function or
                                            if the evaluations will be saved on disk in the evaluation directory.
    :return:
    """
    # Calculate the number of samples in the validation set
    validation_size = int(math.ceil(float(len(sample_list) * percentage)))
    # Randomly pick samples until %-split percentage
    validation = []
    for i in range(validation_size):
        validation_sample = sample_list.pop(np.random.choice(len(sample_list)))
        validation.append(validation_sample)
    # Rename the remaining cases as training
    training = sample_list
    # Reset Neural Network model weights
    model.reset_weights()
    # Run training & validation
    history = model.evaluate(training, validation, epochs=epochs,
                             iterations=iterations, callbacks=callbacks)
    # Initialize evaluation directory
    create_directories(evaluation_path)
    # Draw plots for the training & validation
    if draw_figures:
        plot_validation(history.history, model.metrics, evaluation_path)
    # Make a detailed validation
    if run_detailed_evaluation:
        detailed_validation(validation, model, evaluation_path)
    # Return or backup the validation results
    if return_output : return history.history
    else : backup_history(history.history, evaluation_path)
