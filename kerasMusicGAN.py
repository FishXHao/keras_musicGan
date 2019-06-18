#%%
#from music21 import converter, instrument, note, chord, midi, stream
from music21 import *
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import random

#%%
def tempoModel(sequence_length, tempo_len, optimizer=Adam(lr=0.0002, beta_1=0.5)):
    """
    model = Sequential()
    model.add(Dense(units=1200, activation='relu'))
    model.add(Dense(units=1200, activation='relu'))
    model.add(Dense(output_dim=1))
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    """
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, tempo_len)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(tempo_len))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model
#%%
notes = []
track = 0
tempoOffset = []
#tempoChange = []
for i, file in enumerate(glob.glob("data/Stranger_In_The_North.mid")):
    try:
        midi = converter.parse(file)
        # There are multiple tracks in the MIDI file, so we'll use the first one
        midi = midi
        print(len(midi))
        notes_to_parse = None

        # Parse the midi file by the notes it contains
        notes_to_parse = midi.flat.notes
            
        for index, element in enumerate(notes_to_parse):
            #print(notes_to_parse.getInstrument())
            #if (element.offset)/4<sum(tempoOffset):
            #tempoChange=[]
            if len(tempoOffset)==0:
                #tempoChange.append(element.offset)
                tempoOffset.append(element.offset)
            else:
                tempoOffset.append(element.offset-sum(tempoOffset))
                #tempoChange.append(element.offset-sum(tempoChange))
                
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # get's the normal order (numerical representation) of the chord
                notes.append('.'.join(str(n) for n in element.normalOrder))
        print("Song {} Loaded".format(i+1))
    except Exception as e:
        print(e)
#print(tempoOffset)
print("DONE LOADING SONGS")
# Get all pitch names
pitches = sorted(set(item for item in notes))
tempoes = sorted(set(item for item in tempoOffset))
# Get all pitch names
vocab_length = len(pitches)  
tempo_len = len(tempoes)
number_notes = len(notes)
print(vocab_length)
#print(notes)

#%%
# Let's use One Hot Encoding for each of the notes and create an array as such of sequences. 
#Let's first assign an index to each of the possible notes
note_dict = dict()
for i, notee in enumerate(pitches):
    note_dict[notee] = i
print(note_dict)
tempo_dict = dict()
for i, notee in enumerate(tempoes):
    tempo_dict[notee] = i
print(tempo_dict)
#%%

sequence_length = 50
# Lets make a numpy array with the number of training examples, sequence length, and the length of the one-hot-encoding
num_training = number_notes - sequence_length

input_notes = np.zeros((num_training, sequence_length, tempo_len))
output_notes = np.zeros((num_training, tempo_len))

for i in range(0, num_training):
    # Here, i is the training example, j is the note in the sequence for a specific training example
    input_sequence = tempoOffset[i: i+sequence_length]
    output_note = tempoOffset[i+sequence_length]
    for j, notee in enumerate(input_sequence):
        input_notes[i][j][tempo_dict[notee]] = 1
    output_notes[i][tempo_dict[output_note]] = 1
"""
tempoX = np.array(tempoOffset[0:-1])
tempoX = tempoX.reshape((-1,1))
tempoY = np.array(tempoOffset[1:])
tempoY = tempoY.reshape((-1,1))
"""

backtempo_dict = dict()
for notee in tempo_dict.keys():
    index = tempo_dict[notee]
    backtempo_dict[index] = notee
print(backtempo_dict)

tempomodel = tempoModel(sequence_length,tempo_len)
tempomodel.fit(input_notes, output_notes, batch_size=128, nb_epoch=150)
tempomodel.save('tempoModel.h5')

#tempomodel = tf.contrib.keras.models.load_model('tempoModel.h5')
# pick a random sequence from the input as a starting point for the prediction
n = np.random.randint(0, len(input_notes)-1)
sequence = input_notes[n]
start_sequence = sequence.reshape(1, sequence_length, tempo_len)
output = []
tempo = []
# Let's generate a song of 100 notes
for i in range(0, 700):
    newNote = tempomodel.predict(start_sequence, verbose=0)
    # Get the position with the highest probability
    index = np.argmax(newNote)
    encoded_note = np.zeros((tempo_len))
    encoded_note[index] = 1
    tempo.append(tempoOffset[index]) 
    output.append(encoded_note)
    sequence = start_sequence[0][1:]
    start_sequence = np.concatenate((sequence, encoded_note.reshape(1, tempo_len)))
    start_sequence = start_sequence.reshape(1, sequence_length, tempo_len)
finaltempo = [] 
for element in output:
    index = list(element).index(1)
    finaltempo.append(backtempo_dict[index])
print(finaltempo)
#%%
# Now let's construct sequences. Taking each note and encoding it as a numpy array with a 1 in the position of the note it has
sequence_length = 50
# Lets make a numpy array with the number of training examples, sequence length, and the length of the one-hot-encoding
num_training = number_notes - sequence_length

input_notes = np.zeros((num_training, sequence_length, vocab_length))
output_notes = np.zeros((num_training, vocab_length))

for i in range(0, num_training):
    # Here, i is the training example, j is the note in the sequence for a specific training example
    input_sequence = notes[i: i+sequence_length]
    output_note = notes[i+sequence_length]
    for j, notee in enumerate(input_sequence):
        input_notes[i][j][note_dict[notee]] = 1
    output_notes[i][note_dict[output_note]] = 1

#%%
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

"""
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, vocab_length)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(vocab_length))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
history = model.fit(input_notes, output_notes, batch_size=128, nb_epoch=400)

model.save('musicGanModel2.h5')

#%%
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
"""
model = tf.contrib.keras.models.load_model('musicGanModel2.h5')

#%%
# Make a dictionary going backwards (with index as key and the note as the value)
backward_dict = dict()
for notee in note_dict.keys():
    index = note_dict[notee]
    backward_dict[index] = notee

# pick a random sequence from the input as a starting point for the prediction
n = np.random.randint(0, len(input_notes)-1)
sequence = input_notes[n]
start_sequence = sequence.reshape(1, sequence_length, vocab_length)
output = []
tempo = []
# Let's generate a song of 100 notes
for i in range(0, 700):
    newNote = model.predict(start_sequence, verbose=0)
    # Get the position with the highest probability
    index = np.argmax(newNote)
    encoded_note = np.zeros((vocab_length))
    encoded_note[index] = 1
    tempo.append(tempoOffset[index]) 
    output.append(encoded_note)
    sequence = start_sequence[0][1:]
    start_sequence = np.concatenate((sequence, encoded_note.reshape(1, vocab_length)))
    start_sequence = start_sequence.reshape(1, sequence_length, vocab_length)
    
# Now output is populated with notes in their string form
#for element in output:
    #print(element)

#%%
finalNotes = [] 
for element in output:
    index = list(element).index(1)
    finalNotes.append(backward_dict[index])
    
offset = 0
output_notes = []
    
# create note and chord objects based on the values generated by the model
for i, pattern in enumerate(finalNotes):
    # pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            #new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # pattern is a note
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        #new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    # increase offset each iteration so that notes do not stack
    offset += finaltempo[i]
output_notes.insert(0, instrument.KeyboardInstrument())



# Now output is populated with notes in their string form
#for element in output:
    #print(element)

#%%
offset = 0
out_notes = []
    
# create note and chord objects based on the values generated by the model
for i, pattern in enumerate(finalNotes):
    # pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            #new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
            break
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        out_notes.append(new_chord)
    # pattern is a note
    else:
        pass
        """
        new_note = note.Note(pattern)
        new_note.offset = offset
        #new_note.storedInstrument = instrument.Piano()
        out_notes.append(new_note)
        """

    # increase offset each iteration so that notes do not stack
    offset += finaltempo[i]
out_notes.insert(0, instrument.SteelDrum())
output_notes=stream.Stream(output_notes)
out_notes=stream.Stream(out_notes)
midi_stream = stream.Stream([output_notes, out_notes])
#midi.getInstruments().show('text')
#midi_stream = stream.Stream(output_notes)
"""
#get all instruments
parts = instrument.partitionByInstrument(mid)
for i in parts:
    print(i)
"""
#print(midi.show('text'))
#print(midi_stream.elements)
midi_stream.write('midi', fp='test_output.mid')
