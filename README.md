
# ChessMix: Spatial Context Data Augmentation for Remote Sensing Semantic Segmentation

------------

## Chessmix usage instructions:

Hyperparameters defined in the second cell of the Jupyter Notebook files:
- **crop_size** = size of the final synthetic images
- **patch_size** = lowerst size of the mini-patches that will compose synthetic images
- **n_scale** = number of scaling factors for the mini-patches' size
- **patch_overlap** = percentage of overlap between neighbor mini-patches across the whole dataset 
- **n_classes** = number of classes of the dataset

Instructions:
1) Define the hyperparameters following the descriptions above.
2) In the third cell, set the name of the dataset folder (dataset_name). Inside the dataset folder, it is expected to be two folder: "images" and "masks" (each one with the corresponding data and with the label presenting the same name as the corresponding image).
3) Set the variable ignore_set with the list of the name of the images that will be ignored due to taking part of the validation or test sets.
4) Run the rest of the cell. The new synthetic images and the corresponding labels will be saved in the "new_data" folder. The "n_images" variable defines the amount of images to be saved.

For the unlabeled parts of images (from the black squares of ChessMix's images or in datasets with unlabed pixels), we set the pixel value 0 and the rest of the labeled classes will be from 1 onward. Later on the training code, these pixels will be subtracted by one, meaning the unlabeled pixels will be -1 (the number chosen to be the ignore_index of the function) and the rest of the classes from 0 onward.

------------

## Training and test instructions:

By default, we expect the final dataset of original+synthetic images to follow the same format of the dataset folder used in the ChessMix notebooks', which means there should be a folder named images (with the images) and a folder masks (with the labels, following the same name as the corresponding images). Furthermore, we expect the original images' names to start with a non-digit character, as the new synthetic images are (by default) named 1.png, 2.png and so on. There should also be files named "train.txt", "val.txt" and "test.txt" in the root of the dataset folder, containing the name of the images composing each one of corresponding data divisions (one image per line of the file).

Examples for how to perform the training and test procedures are present in the train_test.sh file. Follow the args definition in main and test files for details about the parameters.
