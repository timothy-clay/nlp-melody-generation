import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from melody_utils import START_MELODY, END_MELODY

class TransformersDataset(Dataset):
    """ 
    A custom dataset to store melody data for the GPT model
    """
    def __init__(self, encoded_melodies, ngram):
        """
        Defines initialization strategy for a TransformersDataset object, treating the input essentially like ngrams.

        Inputs:
            encoded_melodies (list): A list of lists of numeric representations of notes and rhythms to be stored
            ngram (int): How long each sequence in the TransformersDataset should be
        """

        self.examples = []

        # loop through each melody
        for melody in encoded_melodies:
            
            # go through the melody token-by-token in a moving window of size ngram
            for i in range(0, len(melody) - (ngram + 1)):
                
                # define input sequence as a window of size ngram and the label sequence as the same window, but shifted by one
                input_sequence = melody[i:i+ngram]
                label_sequence = melody[i+1:i+ngram+1]
                            
                # add input_ids and labels to the examples
                self.examples.append({
                        "input_ids": torch.tensor(input_sequence),
                        "labels": torch.tensor(label_sequence),
                    })
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

if __name__ == "__main__":
    pass

class GPT2:

    def __init__(self, dataset, encoder, n_positions=128, n_ctx=128, n_embd=128, n_layer=4, n_head=4):
        """
        Creates an untrained GPT2 model with specified parameters.

        Inputs:
            dataset (TransformersDataset): A dataset on which the model should eventually be trained on
            encoder (note_encoder): A note encoder object that can translate between indices and items
            n_positions, n_ctx, n_embd, n_layer, n_head (int): optional parameters to pass through the GPT2 configuration
        """

        # store variables
        self.dataset = dataset
        self.encoder = encoder
        self.n_positions=n_positions
        self.n_ctx=n_ctx
        self.n_embd=n_embd
        self.n_layer=n_layer
        self.n_head=n_head

        # create GPT2Config object using stored variables
        self.gpt_config = GPT2Config(
            vocab_size=len(self.encoder.index_to_item),
            n_positions=self.n_positions,
            n_ctx=self.n_ctx,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head
        )

        # create model using the GP2Config defined above
        self.model = GPT2LMHeadModel(self.gpt_config)

    def _collate(self, batch):
        """
        Specifies how the transformer should collate different observations in a single batch. 

        Inputs:
            batch: a batch of data

        Outputs:
            a dictionary that combines all the input ids and labels into one
        """

        # extract input_ids and labels from the batch
        input_ids = [x["input_ids"] for x in batch]
        labels = [x["labels"] for x in batch]

        # pad the input_ids and labels to be the same lengths
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=-100)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {"input_ids": input_ids, "labels": labels}

    def train(self, per_device_train_batch_size=4, logging_steps=100, save_strategy="no", report_to="none", 
                    max_steps=-1, data_collator=None, epochs=1):
        """
        Trains the model using defined training parameters.

        Inputs:
            per_device_train_batch_size, num_train_epochs, logging_steps, save_strategy, report_to, max_steps, data_collator, 
                epochs:  parameters for the TrainingArguments object

        Outputs:
            none
        """
        
        # sets the data collator to be the one defined previously if none is provided
        if not data_collator:
            data_collator = self._collate
        
        # creates a TrainingArguments object
        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            report_to=report_to,
            max_steps=max_steps,
            num_train_epochs=epochs
        )

        # creates a Traner object using the GPT2 model, the TrainingArguments object, the dataset, and the collator
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator
        )

        # trains the model using the specified trainer
        trainer.train()

    def generate_melody(self, max_notes):
        """
        Generates a melody of max_notes length by using the GPT2 model's built-in generate() function.

        Inputs:
            max_notes (int): how long the generated melody should be

        Outputs:
            generated (list): a list of strings that, when put together, produce a simple melody
        """

        # initialize the an empty input list and attention mask
        input_ids = torch.tensor([[self.encoder.item_to_index[START_MELODY]]])
        attention_mask = torch.tensor([[1]])
        
        # send the model and the inputs to the cpu
        self.model = self.model.to("cpu")
        input_ids = input_ids.to("cpu")

        # generates until max_notes has been hit
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length = max_notes,
            eos_token_id=None,           
            num_return_sequences=1,  
            do_sample=True
        )

        # removes the START_MELODY token from the generated tokens and decodes the indices back into items
        generated_tokens = [self.encoder.index_to_item[encoded] for encoded in generated_ids.tolist()[0] if self.encoder.index_to_item[encoded] != START_MELODY]

        # split notes/rhythms into separate items in the list and flatted said list out
        generated_tokens = [note_item for note_pair in generated_tokens for note_item in note_pair.split('-')]
        
        return generated_tokens