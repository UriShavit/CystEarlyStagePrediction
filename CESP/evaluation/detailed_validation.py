import numpy as np
from CESP.utils.visualizer import visualize_evaluation
from CESP.data_loading.data_io import backup_evaluation


def detailed_validation(validation_samples, model, evaluation_path):
    """
    Function for detailed validation of a validation sample data set. The segmentation
    of these samples will be predicted with an already fitted model and evaluated.
    :param validation_samples:
    :param model: Instance of an already fitted Neural Network model class instance.
    :param evaluation_path: used for storing all kinds of evaluation results during the validation processes.
                                            if the evaluations will be saved on disk in the evaluation directory.
    :return:
    """
    # Initialize detailed validation scoring file
    classes = list(map(lambda x: "dice_class-" + str(x),
                       range(model.preprocessor.data_io.interface.classes)))
    header = ["sample_id"]
    header.extend(classes)
    backup_evaluation(header, evaluation_path, start=True)
    # Iterate over each sample
    for sample_index in validation_samples:
        # Predict the sample with the provided model
        model.predict([sample_index], return_output=False)
        # Load the sample
        sample = model.preprocessor.data_io.sample_loader(sample_index,
                                                          load_seg=True,
                                                          load_pred=True)
        # Access image, truth and predicted segmentation data
        img, seg, pred = sample.img_data, sample.seg_data, sample.pred_data
        # Calculate classwise dice score
        dice_scores = compute_dice(seg, pred, len(classes))
        # Save detailed validation scores to file
        scores = [sample_index]
        scores.extend(dice_scores)
        backup_evaluation(scores, evaluation_path, start=False)
        # Visualize the truth and prediction segmentation
        visualize_evaluation(sample_index, img, seg, pred, evaluation_path)


def compute_dice(truth, pred, classes):
    """
    Calculates class-wise dice similarity coefficient
    :param truth:
    :param pred:
    :param classes:
    :return:
    """
    dice_scores = []
    # Compute Dice for each class
    for i in range(classes):
        try:
            pd = np.equal(pred, i)
            gt = np.equal(truth, i)
            dice = 2*np.logical_and(pd, gt).sum()/(pd.sum() + gt.sum())
            dice_scores.append(dice)
        except ZeroDivisionError:
            dice_scores.append(0.0)
    # Return computed Dice scores
    return dice_scores
