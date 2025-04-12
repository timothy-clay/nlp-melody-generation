import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchinfo import summary
from sklearn.model_selection import train_test_split
import pickle
from melody_utils import START_MELODY, END_MELODY, NOTE_LABELS, RHYTHM_LABELS


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder, ngram, hidden_units=128):
        """
        Initialize a new untrained model. 
        
        Inputs:
            vocab_size (int): The number of tokens in the vocabulary
            embedding_dim (int): The dimension of the embeddings that the model should create
            encoder (note_encoder): A note encoder object that can translate between indices and items
            ngram (int): The value of n for training and prediction
            hidden_unit (int): The size of the hidden layer

        Outputs: 
            none
        """        

        super().__init__()

        # store encoder and ngram values
        self.encoder = encoder
        self.ngram = ngram

        # create model architecture
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.hidden_units = hidden_units
        self.rnn = nn.RNN(embedding_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, vocab_size)
    
    def forward(self, X):
        """
        Compute the forward pass through the network.

        Inputs:
            X: the input data

        Ouputs:
            The output of the model; i.e. its predictions.
        
        """

        # pass X through each step in the architecture
        embedded_X = self.embedding(X)
        output, _ = self.rnn(embedded_X)
        last_output = output[:, -1, :]
        result = self.fc(last_output)
        return result

    def fit(self, dataloader, epochs, lr=0.001):
        """
        Fit the RNN to a specific dataset by running it for a specified number of epochs.

        Inputs:
            dataloader: The data that the RNN should be trained on
            epochs (int): The number of times that the training loop should be run
            lr (float): The learning rate for gradient descent
        """

        # initalize loss function and the optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):

            total_loss = 0

            # loop through all data stored in the dataloader
            for step, data in enumerate(dataloader):

                # extract inputs and labels (X and y)
                inputs, labels = data

                # zero out the gradients
                optimizer.zero_grad()

                # generate predictions from the inputs and calculate loss of said predictions
                pred = self(inputs)
                loss = loss_fn(pred, labels)

                # back-prop the loss
                loss.backward()

                # gradient descent step
                optimizer.step()

                # add the loss to the total loss
                total_loss += loss.item()

            # print the total loss after each epoch
            print(f'Epoch [{epoch + 1}/{epochs}] loss: {total_loss:.4f}')

    def predict(self, input_tokens):
        """
        Generate a predicted next token for a provided list of tokens.

        Inputs:
            input_tokens (list): a list of previous tokens in the melody that should be used to generate the next one

        Outputs:
            prediction: a string representation of the predicted token (e.g. 'C' or '60' or 'whole' etc.)
        """
    
        self.eval()  
        
        # convert input tokens to their corresponding index
        encoded_tokens = [self.encoder.item_to_index[token] for token in input_tokens]

        # get inputs and outputs from the model
        input = torch.tensor([encoded_tokens])
        output = self(input)

        # get the probabilities for each output by applying the softmax function
        probs = nn.functional.softmax(output, dim=1)

        # choose randomly from the tokens weighted by the probabilities
        pred_index = torch.multinomial(probs, 1).item()

        # convert index back to token and return it
        return self.encoder.index_to_item[pred_index]

    def generate_melody(self, max_notes):
        """
        Generates a melody of max_notes length by repeatedly calling the predict function on what starts as an empty list.

        Inputs:
            max_notes (int): how long the generated melody should be

        Outputs:
            generated (list): a list of strings that, when put together, produce a simple melody
        """

        # initialization
        generated = [START_MELODY] * (self.ngram - 1)
        last_generated = ''

        # loop until the melody is max_notes notes long
        while len(generated) - (self.ngram - 1) <= (max_notes * 2) + 1:

            # predict and store the next token
            new_generated = self.predict(generated[(len(generated) - self.ngram)+1:])

            # prevents a sentence begin or end token from being generated
            if new_generated in [START_MELODY, END_MELODY]:
                continue

            # if last generated was note, must be a rythym
            if last_generated != '' and last_generated not in RHYTHM_LABELS and new_generated not in RHYTHM_LABELS:
                continue

            # if last generated was a rhythm, must be a note or end of melody
            if last_generated in RHYTHM_LABELS + [START_MELODY] and new_generated in RHYTHM_LABELS:
                continue

            # update the last generated token
            last_generated = new_generated

            # add the last generated token to the generated list
            generated.append(last_generated)

        # removes start tokens
        generated = [token for token in generated if token != START_MELODY]

        return generated



if __name__ == '__main__':
    pass