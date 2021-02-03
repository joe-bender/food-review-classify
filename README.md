# Character-Based Food Review Sentiment Classification

I created a sentiment classifier using [fine food reviews from Amazon](https://www.kaggle.com/snap/amazon-fine-food-reviews) and a character-based model built using only PyTorch and Pandas. 

## Features
- a batching method that uses both sorting and randomness to create batches that never need excessive amounts of padding but also are somewhat random
- a character-based LSTM model that averages its full sequence of outputs instead of only using the final output, for better prediction
- character embedding layer in model to learn meaningful representations of each character and deal with capitalization effectively
- tokenization and numericalization from scratch
- dataset and batching functionality from scratch
- training and validation loops from scratch
- prediction and analysis functions from scratch


The main goal of this project was to create a batching method that balances the need for randomness between epochs with the need to keep items with similar lengths in each batch. The secondary goal was to train a sentiment classifier with characters instead of words as tokens. 

I achieved 91% accuracy on the validation set after just 4 epochs, which means the model and surrounding functions worked very well for this task. I also examined the full sequence of outputs at the end of the notebook for a single example prediction, which is an interesting visualization of how the model continues to change its mind after encountering each character of the review text.

## Running this project on your machine

This project has only one notebook, which sets up and handles everything. You only need to have the Kaggle api installed along with PyTorch and Pandas.

The training section of the notebook is broken into 4 cells, for each epoch of the training process. Please be aware that each of these cells can take a while to run, depending on your GPU.
