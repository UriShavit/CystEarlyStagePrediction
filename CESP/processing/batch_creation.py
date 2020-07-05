
import math
import numpy as np


def create_batches(img_queue, batch_size, incomplete_batches=True, last_index=False):
    """
    Create batches from a queue of preprocessed images
    :param img_queue:
    :param batch_size:
    :param incomplete_batches:
    :param last_index:
    :return:
    """
    # Calculate the number of batches for the current image queue
    batch_number = math.floor(len(img_queue) / batch_size)
    # Gather images to batches
    batches = []
    for i in range(0, batch_number):
        # Identify images which will be relocated in the next batch
        start = i * batch_size
        end = start + batch_size
        # Gather images and combine them into a batch
        try : batch = collect_batch(img_queue, start, end)
        except : batch = None
        # Add batch to finished batches list
        if batch is not None : batches.append(batch)
    # Handle remaining images in the image queue
    if len(img_queue) % batch_size != 0:
        # Create a batch which is smaller than provided batch size
        if incomplete_batches:
            # Identify images which will be relocated in the next batch
            start = batch_number * batch_size
            end = len(img_queue)
            # Gather images and combine them into a batch
            try : batch = collect_batch(img_queue, start, end)
            except : batch = None
            # Add batch to finished batches list
            if batch is not None : batches.append(batch)
        # Stock up the last batch in the epoch with already batched images
        elif last_index:
            # Identify images which will be relocated in the next batch
            end = len(img_queue)
            if (end - batch_size) >= 0 : start = end - batch_size
            else : start = 0
            # Gather images and combine them into a batch
            try : batch = collect_batch(img_queue, start, end)
            except : batch = None
            # Add batch to finished batches list
            if batch is not None : batches.append(batch)
    # Update image queue by removing already batched images
    update_queue(img_queue, batch_size, incomplete_batches, last_index)
    # Return list of created batches
    return batches


def collect_batch(img_queue, start, end):
    """
    Subroutines gathers images and combine them into a batch
    :param img_queue:
    :param start:
    :param end:
    :return:
    """
    # Iterate over the images which will be relocated in a batch
    img_list = []
    seg_list = []
    for j in range(start, end):
        # Access these images
        img = img_queue[j][0]
        if len(img_queue[j]) == 2 : seg = img_queue[j][1]
        else : seg = None
        # Add images to associated list
        img_list.append(img)
        seg_list.append(seg)
    # Combine images into a batch
    batch_img = np.stack(img_list, axis=0)
    if any(elem is None for elem in seg_list) : batch_seg = None
    else : batch_seg = np.stack(seg_list, axis=0)
    # Combine batch_img and batch_seg into a tuple
    batch = (batch_img, batch_seg)
    # Return finished batch
    return batch


def update_queue(img_queue, batch_size, incomplete_batches=False, last_index=False):
    """
    Update image queue by removing already batched images
    :param img_queue:
    :param batch_size:
    :param incomplete_batches:
    :param last_index:
    :return:
    """
    # Clean complete image queue if incomplete batches are allowed or last index
    if incomplete_batches or last_index:
        img_queue.clear()
    # Delete only images from the image queue which were already batched
    else:
        # Identify the end position of the last image which was batched
        batch_number = math.floor(len(img_queue) / batch_size)
        end = batch_number * batch_size
        # Delete batched images from image queue
        del img_queue[0:end]
