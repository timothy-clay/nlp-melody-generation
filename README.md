# Natural Language Processing Final Project: Melody Generation
Timothy Clay

## Overview
This repository contains all the source code I used to create three language models ($n$-gram, RNN, and GPT-2) for the purpose of generating a simple melody. 

## File Descriptions
This repository contains 8 files: 1 .ipynb notebook, 6 .py files, and 1 .pdf (the project's written report). Each file is described in more detail below:

- `melody_generation.ipynb`: This file is the central/main file, which walks through the set-up, training, and evaluation steps for each of the three models. It frequently calls upon custom-defined functions contained in the remaining .py files. This file overlaps significantly with the project's written report, but should not be viewed as a replacement for this report, but instead as a supplement.
- `melody_utils.py`: This file contains a variety of miscellaneous utility functions that I used throughout the project. Functions include: how to scrape raw data, how to tokenize a melody, and how to write a melody to a MIDI file, among others.
- `note_encoder.py`: This file contains the definition for a `note_encoder` object, which is a helpful object I used to encode and decode notes and rhythms into their respective integer representations.
- `ngram.py`: This file contains an `NGram` class, which is used to both make and train an $n$-gram model, as well as generate a melody of a certain length using said model. This file also contains a `create_ngrams()` function, which, as the name suggests, creates $n$-grams from a list of tokens.
- `rnn.py`: This file contains an `RNN` class, which is used to both make and train an RNN model, as well as generate a melody of a certain length using said model.
- `tranformer.py`: This file contains an `GPT2` class, which is used to fine-tune a GPT-2 model using the tools provided from the `transformers` library. This class also is able to generate a melody of a certain length using said trained model.
