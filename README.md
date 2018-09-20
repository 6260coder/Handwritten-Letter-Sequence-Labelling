Purpose of model:  
Classify each member in a sequence taking into acount information from time steps before and after it.

Structure of model:  
two-directional recurrent neural network with shared softmax across each time step for classification

Management of files:  
* **data_helpers.py** holds utility functions regarding data preprocessing.
* **sequence_labelling_model_bidirectional.py** holds the structure of the neural network as a class.
* **Train.py** implements the training of the network.

Dataset:  
The OCR dataset collected by Rob Kassel at the MIT Spoken Language Systems Group and preprocessed by Ben Taskar at the Stanford AI group. The dataset contains individual hand-written letters as binary images of 16 times 8 pixels. The letters are arranged into sequences that where each sequence forms a word. In the whole dataset, there are about 6800 words of length up to 14.   
The dataset is a csv file held in a gzip archive.  
The dataset is available at http://ai.stanford.edu/~btaskar/ocr/.  
After preprocessing:  
shape of labels: (6877, 14, 26)  
shape of sequences: (6877, 14, 128)
