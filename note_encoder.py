class note_encoder:
    def __init__(self):
        """
        Creates untrained encoder object

        Inputs:
            none
        
        Outputs:
            none
        """

        # stores vocab and vocab size
        self.note_items = set()
        self.vocab_size = 0

        # stores items to indices (i.e. 'C' -> 1, 'whole' -> 2, etc.)
        self.item_to_index = {}

        # stores indices to items (i.e. 1 -> 'C', 2 -> 'whole', etc.)
        self.index_to_item = {}

    def fit(self, melodies):
        """
        Fits the note encoder object to a list of melodies (i.e. creates encoding strategy)

        Inputs:
            melodies (list): a list of lists containing notes and rhythms. The tokenization of the inner melodies can differ 
                             for each note_encoder object


        """

        # add each item that appears to the vocab set
        for melody in melodies:
            for item in melody:
                self.note_items.add(item)

        # create pairings between indices and items
        self.item_to_index = {item: idx for idx, item in enumerate(self.note_items)}
        self.index_to_item = {idx: item for idx, item in enumerate(self.note_items)}

        # get vocab size my measuring the length of the item_to_index dictionary
        self.vocab_size = len(self.item_to_index)

    def transform(self, melodies):
        """
        Transforms the provided data using the encoding strategy defined by the model's fit function.

        Inputs:
            melodies (list): a list of lists containing notes and rhythms to be transformed. The tokenization of the inner melodies 
                             can differ for each note_encoder object

        Outputs:
            encoded_melodies (list): a list of lists of the same shape as the input melodies, but this time encoded
        """

        # recreate the input melodies list of lists by passing each item through self.item_to_index
        encoded_melodies = []
        for melody in melodies:
            encoded_melodies.append([self.item_to_index[note] for note in melody])
        return encoded_melodies

    def fit_transform(self, melodies):
        """
        Fits the note_encoder object and transforms the same data at the same time.

        Inputs:
            melodies (list): a list of lists containing notes and rhythms. The tokenization of the inner melodies can differ 
                             for each note_encoder object

        Outputs:
            encoded_melodies (list): a list of lists of the same shape as the input melodies, but this time encoded
        """

        # fit and then transform the melodies
        self.fit(melodies)
        return self.transform(melodies)

if __name__ == '__main__':
    pass