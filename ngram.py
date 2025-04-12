from collections import Counter
import random
from melody_utils import START_MELODY, END_MELODY, NOTE_LABELS, RHYTHM_LABELS

def create_ngrams(tokens, n, glue=False):
  """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    ngrams (list): either a list of lists of tokens (if glue=False) or a list of strings of the same tokens, concatenated using a 
                   space (if glue=True)
  """

  ngrams = []
  
  # use a moving window of size n to compute and store all n-grams in the tokens list
  for i in range(len(tokens) - n + 1):

    ngram = tokens[i : i+n]

    # if glue is True, concatenate all tokens using a space
    if glue:
      ngram = " ".join(ngram)

    ngrams.append(ngram)
  
  return ngrams

class NGram:
  def __init__(self, n_gram):
    """
    Initializes an untrained NGram language model
    
    Inputs:
      n_gram (int): the n-gram order of the model to create

    Outputs:
      none
    """

    # ensure ngram size (so that at least a full note is being considered at all times)
    assert n_gram > 1, "NGram must be >= 2"

    self.n_gram = n_gram

    # create data structures to store the unique terms and how many times they appear in the training data
    self.vocab_counts = Counter()
    self.vocab = set()

    # create data structures to store the number of times each ngram and each subgram (ngram of size n-1) occurs
    self.ngrams = Counter()
    self.subgrams = Counter()

  def train(self, tokenized_melodies):
    """
    Trains the language model on the given data. Assumes that the given data has lists of tokens that have already been processed.
    
    Inputs:
      tokenized_melodies (list): a list that contains sublists of tokenized melodies
          e.g. [['C', 'whole', 'B', 'half', ...], ['A', 'eighth', 'D', 'dotted_quarter', ...], ...]

    Outputs: 
      none
    """

    for melody in tokenized_melodies:

        # count the number of times that each token appears in the melody
        for token in melody:
            self.vocab_counts[token] += 1

        # goes through each melody and adds them to the vocab set
        for token in melody:
            self.vocab.add(token)

        # creates ngrams and subgrams and stores them
        for ngram in create_ngrams(melody, self.n_gram, glue=True):
              self.ngrams[ngram] += 1
              self.subgrams[ngram[:-1]] += 1

    return

  def score(self, tokens, laplace=True):
    """
    Calculates the probability score for a given string representing a single sequence of tokens.
    
    Inputs:
      tokens (list): a tokenized sequence to be scored by this model
      
    Outputs:
      num / den (float): the probability value of the given tokens for this model
    """
    
    # converts input tokens into ngrams
    ngrams = create_ngrams(tokens, self.n_gram, glue=True)

    # initialization
    num = 1
    den = 1

    for ngram in ngrams:

      # updates the numerator 
      if laplace:
        num *= self.ngrams[ngram] + 1    # adds 1 to numerator if using laplace smoothing
      else:
        num *= self.ngrams[ngram]    # does not add anything to numerator if not using laplace smoothing

      # updates the denominator
      if laplace:
        den *= self.subgrams[ngram[:-1]] + len(self.subgrams)   # adds number of subgrams to denominator if using laplace smoothing
      else:
        den *= self.subgrams[ngram[:-1]]     # does not add anything to denominator if not using laplace smoothing
        
    return num / den

  def generate_melody(self, max_notes) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    
    # creates starting sequence for the melody of MELODY_START tokens and defaults the last_generated value to the same token
    melody = [START_MELODY for n in range(self.n_gram-1)]
    last_generated = START_MELODY

    # loops until the model outputs the maximum number of generated notes
    while len(melody) <= (max_notes * 2) + self.n_gram:

      # stores the probability of each new token being the next token
      token_probs = {}
      for token in self.vocab:
        
        # prevents a sentence begin or end token from being generated
        if token in [START_MELODY, END_MELODY]:
            continue

        # if last generated was note, must be a rhythm
        if last_generated in NOTE_LABELS and token not in RHYTHM_LABELS:
            continue

        # if last generated was a rhythm, must be a note or end of melody
        if last_generated in RHYTHM_LABELS + [START_MELODY] and token in RHYTHM_LABELS:
            continue

        # adds token to end of existing sentence
        gen_tuple = melody[-(self.n_gram - 1):] + [token]
          
        # calculates what the score would be of the generated tuple
        token_probs[token] = self.score(gen_tuple, laplace=True)

      # extracts the tokens and probabilities from the token_probs dict
      tokens = list(token_probs.keys())
      probs = list(token_probs.values())

      # samples the tokens using the probabilites and chooses the next generated word
      last_generated = random.choices(tokens, weights=probs)[0]
      melody.append(last_generated)

    # removes start tokens
    melody = [token for token in melody if token != START_MELODY]

    return melody

if __name__ == '__main__':
  pass

