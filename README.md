# Natural Language Processing Final Project: Melody Generation
Timothy Clay

Last Updated: April 13th, 2025

## Overview
This repository contains all the source code I used to create three language models ($n$-gram, RNN, and GPT-2) for the purpose of generating a simple melody. 

## File Descriptions
This repository contains 7 files: 1 .ipynb notebook, 5 .py files, and 1 .pdf (the project's written report). Each file is described in more detail below:

- `melody_generation.ipynb`: This file is the central/main file, which walks through the set-up, training, and evaluation steps for each of the three models. It frequently calls upon custom-defined functions contained in the remaining .py files. This file overlaps significantly with the project's written report, but should not be viewed as a replacement for this report, but instead as a supplement.
- `melody_utils.py`: This file contains a variety of miscellaneous utility functions that I used throughout the project. Functions include: how to scrape raw data, how to tokenize a melody, and how to write a melody to a MIDI file, among others.
- `note_encoder.py`: This file contains the definition for a `note_encoder` object, which is a helpful object I used to encode and decode notes and rhythms into their respective integer representations.
- `ngram.py`: This file contains an `NGram` class, which is used to both make and train an $n$-gram model, as well as generate a melody of a certain length using said model. This file also contains a `create_ngrams()` function, which, as the name suggests, creates $n$-grams from a list of tokens.
- `rnn.py`: This file contains an `RNN` class, which is used to both make and train an RNN model, as well as generate a melody of a certain length using said model.
- `tranformer.py`: This file contains an `GPT2` class, which is used to fine-tune a GPT-2 model using the tools provided from the `transformers` library. This class also is able to generate a melody of a certain length using said trained model.
- `report.pdf`: The final written report for this project. 

## Directions
To run the main `melody_generation.ipynb` file, there are several external files you will need to download. If interested in training the models, you will only need to download the pickled data used in the first cell of code. If interested in using the models without training them, you will need to download the pickled versions of the trained RNN and GPT-2 models (the $n$-gram model did not take long enough to train to justify pickling). If you want to display the training losses, you will need to download the .csv files that those plots are generated using. Finally, if you want to display the embedded MP3 files, you will need to download those files. All of these files are available in [this Google Drive folder](https://drive.google.com/drive/folders/1OHIbn86rI_iKEju6J4rRkTXvv-Ud30Q6?usp=sharing). 
