import os
from tqdm.notebook import tqdm
import pretty_midi
import pickle

NOTE_LENGTHS = {
    4: "whole",
    3: "dotted_half",
    2: "half",
    1.5: "dotted_quarter",
    1: "quarter",
    0.75: "dotted_eighth",
    0.5: "eighth",
    0.25: "sixteenth"
}

NOTE_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "rest"]
RHYTHM_LABELS = ["whole", "dotted_half", "half", "dotted_quarter", "quarter", "dotted_eighth", "eighth", "sixteenth"]

START_MELODY = '<s>'
END_MELODY = '<e>'


def find_melodies():
    """
    Crawls a file structure that includes MIDI files to find and extract all parts that are labeled as a "melody"

    Inputs:
        none

    Outputs:
        melodies (list[dict]): contains each part labeled "melody" found in the file structure and it's corresponding bpm
    """

    melodies = []
        
    # find and stores all valid file directories in the provided directory ('data')
    filepaths = []
    for root, dirs, files in os.walk('data'):
        dirs[:] = [d for d in dirs if not d.startswith('.')] # skip hidden folders
        
        for file in files:
            if not file.startswith('.'): # skip hidden files
                filepaths.append(os.path.join(root, file))  # add full directory

    # process each filename one-by-one
    for filename in tqdm(filepaths):

        try:
            # load midi file using PrettyMIDI
            midi_data = pretty_midi.PrettyMIDI(filename)
            
            # throw out the file if there are tempo changes (otherwise, note lengths would be inconsistent)
            if midi_data.get_tempo_changes()[0].shape[0] > 1:
                continue
            
            # loop through each track and only add it to the instruments if the track has the name "melody"
            for instrument in midi_data.instruments:
                if "melody" in instrument.name.lower():
                    melodies.append({'bpm':midi_data.get_tempo_changes()[1][0], 'track':instrument})

        # allow user to stop program if needed
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        
        # throw out any files with bad reads
        except Exception:
            continue

    return melodies


def process_track(instrument, bpm, normalize_octaves=False):
    """
    Takes a MIDI instrument track and extracts the notes and rhythms.

    Inputs:
        instrument: a PrettyMIDI Instrument object that is the track to be processed
        bpm (float): the tempo of the MIDI file
        normalize_octaves (bool): whether the processing should differentiate between octaves (60 (C4) or 72 (C5) or simply "C")

    Outputs:
        notes (list): a list of tuples of format (pitch, length) for each note in the part. "pitch" is either a number or a
                      string, depending on the value of normalize_octaves. "length" is a string that appears in RHYTHM_LABELS.
    """

    # initialize
    notes = []
    seconds_per_beat = 60.0 / bpm
    last_end_time = 0 

    # iterate through each note in the track
    for note in instrument.notes:

        # extract start time, compute duration (time), and convert duration to beats
        start_time = note.start
        duration_seconds = note.end - note.start
        duration_beats = duration_seconds / seconds_per_beat 
        
        # checks if there is a rest (longer-that-expected space between notes)
        if start_time - last_end_time > 0.5:

            # gets length of rest
            rest_duration_beats = (start_time - last_end_time) / seconds_per_beat
            closest_rest_length = min(NOTE_LENGTHS.keys(), key=lambda x: abs(x - rest_duration_beats))
            rest_length = NOTE_LENGTHS[closest_rest_length]

            # appends to notes depending on normalize_octaves value
            if normalize_octaves:
                notes.append(('rest', rest_length))
            else:
                notes.append((128, rest_length))

        # converts duration to human-readable text
        closest_length = min(NOTE_LENGTHS.keys(), key=lambda x: abs(x - duration_beats))
        note_length = NOTE_LENGTHS[closest_length]

        # appends to notes depending on normalize_octaves value
        if normalize_octaves:
            notes.append((NOTE_LABELS[note.pitch % 12], note_length))
        else:
            notes.append((note.pitch, note_length))

        # update the end time of the last note
        last_end_time = note.end 

    return notes


def tokenize_melodies(melodies, ngram, glue=False):
    """
    Tokenizes the melodies and prepares them to be split into ngrams. 

    Input:
        melodies (list): a list of lists of tuples of format (pitch, length) for each note in the part. "pitch" is either a number 
                         or a string, depending on the value of normalize_octaves. "length" is a string that appears in 
                         RHYTHM_LABELS. 
        ngram (int): The ngram value that the tokenization strategy should prepare for (how many leading START_MELODY tokens 
                     there should be)
        glue (bool): Whether the note names and rhythms should be tokenized together (i.e. True = 'C-whole', False = 'C', 'whole')
    """

    # rewrites each melody to be of form [note-rhythm, note-rhythm, ...] instead of [(note, rhythm), (note, rhythm), ...]
    concat_melodies = [[str(note) + "-" + length for note, length in melody] for melody in melodies]

    # removes duplicate melodies by combining all items in each melody with a ';' separator, converting to a set and back 
    # to a list, then re-splitting on the ';' separator
    unique_melodies = [ls.split(';') for ls in list(set([';'.join(melody) for melody in concat_melodies]))]

    processed_melodies = []

    # loop through each melody
    for melody in unique_melodies:

        # prepend melody with ngram-1 instances of the START_MELODY token
        current = [START_MELODY] * (ngram - 1)

        # loop through each note in the melody
        for note in melody:

            # if there's a whole note, splits the melody into two separate melodies
            if note in ['128-whole', 'rest-whole']:
                
                if len(current) > (ngram - 1): # unless the whole note is first in the new melody, in which case it is dropped
                    processed_melodies.append(current + [END_MELODY])
                    current = [START_MELODY] * (ngram - 1)
            
            else:

                # adds the concatenate note name and rhythm (i.e. note-rhyhtm) if glue is set to True
                if glue:
                    current += [note]

                # adds the note name and the rhythm separately otherwise
                else:
                    current += note.split('-')

        # if the processed melody is not empty, add it to the processed_melodies list
        if len(current) > 1:
            processed_melodies.append(current + [END_MELODY])
    
    return processed_melodies
     

def write_melody(generated_melody, filename, octaves=True):
    """
    Takes in a list of notes and rhythms and converts it into a MIDI file.

    Inputs:
        generated_melody (list): input notes and rhythms of format [note, rhythm, note, rhythm, ...]
        filename (str): the filename to save the MIDI to
        octaves (bool): whether the note values in generated_melody are numeric (differentiate between octaves) or 
                        string-based (do-not differentiate between octaves)
    
    Outputs:
        none
    """

    # removes tokens that are not notes or rhythms
    notes = [note for note in generated_melody if note not in [START_MELODY, END_MELODY]]

    # invert the NOTE_LENGTHS dictionary to be able to convert length names into numeric lengths
    note_lookup = {note: duration for duration, note in NOTE_LENGTHS.items()}

    # initialize PrettyMIDI object and track
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    current_time = 0.0

    # loop through each note (pair of 2 items) in the generated melody
    for i in range(0, len(notes), 2):

        # extract the note pitch and length
        pitch_str = notes[i]
        duration_name = notes[i + 1]
        duration = note_lookup[duration_name] / 2

        if octaves:

            # catches rests
            if pitch_str != '128':

                # create note object
                pitch = int(pitch_str)
                note = pretty_midi.Note(
                    velocity=100, pitch=pitch,
                    start=current_time,
                    end=current_time + duration
                )

                # add note to track
                instrument.notes.append(note)
            
        else:

            # catches rests
            if pitch_str != 'rest':

                # create note object
                pitch = NOTE_LABELS.index(pitch_str) + 60
                note = pretty_midi.Note(
                    velocity=100, pitch=pitch,
                    start=current_time,
                    end=current_time + duration
                )

                # add note to track
                instrument.notes.append(note)

        # advance time step regardless of if the current item is a note or a rest
        current_time += duration

    # add intrument to file and write the file to the filename
    midi.instruments.append(instrument)
    midi.write(filename)


def load_data(filename):
    """
    Loads a .pkl file. Intended to be used to load previously found raw melodies or previously trained models.
    """
    with open(filename, 'rb') as file:
        content = pickle.load(file)

    return content


def save_data(content, filename):
    """
    Export content to a .pkl file to be loaded later
    """

    with open(filename, 'wb') as file:
        pickle.dump(content, file)


if __name__ == '__main__':
    pass