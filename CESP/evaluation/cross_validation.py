import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import csv
from CESP.data_loading.data_io import create_directories, backup_history
from CESP.utils.plotting import plot_validation
from CESP.evaluation.detailed_validation import detailed_validation


def cross_validation(sample_list, model, k_fold=3, epochs=20,
                     iterations=None, evaluation_path="evaluation",
                     draw_figures=False, run_detailed_evaluation=False,
                     callbacks=[], save_models=True, return_output=False):
    """
    Function for an automatic k-fold Cross-Validation of the Neural Network model by
    running the whole pipeline several times with different data set combinations.

    :param sample_list: A list of sample indicies which will be used for validation.
    :param model: Instance of a Neural Network model class instance.
    :param k_fold: The number of k-folds for the Cross-Validation. By default, a 3-fold Cross-Validation is performed.
    :param epochs: Number of epochs. A single epoch is defined as one iteration through the complete data set.
    :param iterations: Number of iterations (batches) in a single epoch.
    :param evaluation_path: Path to the evaluation data directory. This directory will be created and used for storing
                            all kinds of evaluation results during the validation processes.
    :param draw_figures: Option if evaluation figures should be automatically plotted in the evaluation directory.
    :param run_detailed_evaluation: Option if a detailed evaluation (additional prediction) should be performed.
    :param callbacks: A list of Callback classes for custom evaluation.
    :param save_models: Option if fitted models should be stored or thrown away.
    :param return_output: Option, if computed evaluations will be output as the return of this function or
                            if the evaluations will be saved on disk in the evaluation directory.
    :return:
    """
    # Initialize result cache
    if return_output : validation_results = []
    # Randomly permute the sample list
    samples_permuted = np.random.permutation(sample_list)
    # Split sample list into folds
    folds = np.array_split(samples_permuted, k_fold)
    fold_indices = list(range(len(folds)))
    # Start cross-validation
    for i in fold_indices:
        # Reset Neural Network model weights
        model.reset_weights()
        # Subset training and validation data set
        training = np.concatenate([folds[x] for x in fold_indices if x!=i],
                                  axis=0)
        validation = folds[i]
        # Initialize evaluation subdirectory for current fold
        subdir = create_directories(evaluation_path, "fold_" + str(i))
        # Save model for each fold
        cb_model = ModelCheckpoint(os.path.join(subdir, "model.hdf5"),
                                   monitor="val_loss", verbose=1,
                                   save_best_only=True, mode="min")
        if save_models == True : cb_list = callbacks + [cb_model]
        else : cb_list = callbacks
        # Run training & validation
        history = model.evaluate(training, validation, epochs=epochs,
                                 iterations=iterations, callbacks=cb_list)
        # Backup current history dictionary
        if return_output : validation_results.append(history.history)
        else : backup_history(history.history, subdir)
        # Draw plots for the training & validation
        if draw_figures:
            plot_validation(history.history, model.metrics, subdir)
        # Make a detailed validation of the current cv-fold
        if run_detailed_evaluation:
            detailed_validation(validation, model, subdir)
    # Return the validation results
    if return_output : return validation_results


def split_folds(sample_list, k_fold=3, evaluation_path="evaluation"):
    """
    Function for splitting a data set into k-folds. The splitting will be saved
    in files, which can be used for running a single fold run.
    In contrast to the normal cross_validation() function, this allows running
    folds parallelized on multiple GPUs.
    :param sample_list: A list of sample indicies which will be used for validation.
    :param k_fold: The number of k-folds for the Cross-Validation. By default, a 3-fold Cross-Validation is performed.
    :param evaluation_path:
    :return:
    """
    # Randomly permute the sample list
    samples_permuted = np.random.permutation(sample_list)
    # Split sample list into folds
    folds = np.array_split(samples_permuted, k_fold)
    fold_indices = list(range(len(folds)))
    # Iterate over each fold
    for i in fold_indices:
        # Subset training and validation data set
        training = np.concatenate([folds[x] for x in fold_indices if x!=i],
                                  axis=0)
        validation = folds[i]
        # Initialize evaluation subdirectory for current fold
        subdir = create_directories(evaluation_path, "fold_" + str(i))
        fold_cache = os.path.join(subdir, "sample_list.csv")
        # Write sampling to disk
        write_fold2csv(fold_cache, training, validation)


def run_fold(fold, model, epochs=20, iterations=None, evaluation_path="evaluation", draw_figures=True, callbacks=[],
             save_models=True):
    """
    Function for running a single fold of a cross-validation. In contrast to the normal cross_validation() function,
    this allows running folds parallelized on multiple GPUs.
    :param fold: The integer of the desired fold, which should be validated (starting with 0).
    :param model: Instance of a Neural Network model class instance.
    :param epochs:  Number of epochs. A single epoch is defined as one iteration through the complete data set.
    :param iterations: Number of iterations (batches) in a single epoch.
    :param evaluation_path: Path to the evaluation data directory. This directory will be created and
                            used for storing all kinds of evaluation results during the validation processes.
    :param draw_figures: Option if evaluation figures should be automatically plotted in the evaluation directory.
    :param callbacks: A list of Callback classes for custom evaluation.
    :param save_models: Option if fitted models should be stored or thrown away.
    :return:
    """
    # Load sampling fold from disk
    fold_path = os.path.join(evaluation_path, "fold_" + str(fold),
                             "sample_list.csv")
    training, validation = load_csv2fold(fold_path)
    # Reset Neural Network model weights
    model.reset_weights()
    # Initialize evaluation subdirectory for current fold
    subdir = os.path.join(evaluation_path, "fold_" + str(fold))
    # Save model for each fold
    cb_model = ModelCheckpoint(os.path.join(subdir, "model.hdf5"),
                               monitor="val_loss", verbose=1,
                               save_best_only=True, mode="min")
    if save_models == True : cb_list = callbacks + [cb_model]
    else : cb_list = callbacks
    # Run training & validation
    history = model.evaluate(training, validation, epochs=epochs,
                             iterations=iterations, callbacks=cb_list)
    # Backup current history dictionary
    backup_history(history.history, subdir)
    # Draw plots for the training & validation
    if draw_figures:
        plot_validation(history.history, model.metrics, subdir)


def write_fold2csv(file_path, training, validation):
    """
    Subfunction for writing a fold sampling to disk
    :param file_path:
    :param training:
    :param validation:
    :return:
    """
    with open(file_path, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=" ")
        writer.writerow(["TRAINING:"] + list(training))
        writer.writerow(["VALIDATION:"] + list(validation))

def load_csv2fold(file_path):
    """
    Subfunction for loading a fold sampling from disk
    :param file_path:
    :return:
    """
    training = None
    validation = None
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        for row in reader:
            if not training : training = row[1:]
            else : validation = row[1:]
    return training, validation
