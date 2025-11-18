# Music Generation with AI (LSTM)
import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream, midi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
import pygame
import time

# ---------------------------
# 1️⃣ Load and Preprocess MIDI Files
# ---------------------------

import glob

midi_files = glob.glob(r"C:\Python\midi_data\*.mid")  # Folder with MIDI files
notes = []

for file in midi_files:
    midi_file = converter.parse(file)
    parts = instrument.partitionByInstrument(midi_file)
    if parts:  # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi_file.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print(f"Total notes/chords extracted: {len(notes)}")

# ---------------------------
# 2️⃣ Prepare Sequences for LSTM
# ---------------------------
sequence_length = 50
pitchnames = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(pitchnames)}

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)
print(f"Total sequences for training: {n_patterns}")

X = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(len(pitchnames))
y = np.zeros((n_patterns, len(pitchnames)))
for i, val in enumerate(network_output):
    y[i][val] = 1

# ---------------------------
# 3️⃣ Build LSTM Model
# ---------------------------
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(len(pitchnames)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# ---------------------------
# 4️⃣ Train the Model
# ---------------------------
# Uncomment to train (takes time)
# model.fit(X, y, epochs=100, batch_size=64)

# For demo: load pre-trained weights if available
# model.load_weights("lstm_music_weights.h5")

# ---------------------------
# 5️⃣ Generate Music
# ---------------------------
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]
output_notes = []

for i in range(200):  # generate 200 notes
    prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(len(pitchnames))
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = pitchnames[index]
    output_notes.append(result)
    pattern.append(index)
    pattern = pattern[1:]

# ---------------------------
# 6️⃣ Convert Generated Notes to MIDI
# ---------------------------
offset = 0
output_stream = stream.Stream()

for pattern in output_notes:
    if '.' in pattern or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        chord_notes = [note.Note(int(n)) for n in notes_in_chord]
        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        output_stream.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        output_stream.append(new_note)
    offset += 0.5

midi_file_name = 'generated_music.mid'
output_stream.write('midi', fp=midi_file_name)
print(f"Music generated and saved as {midi_file_name}")

# ---------------------------
# 7️⃣ Play Generated Music
# ---------------------------
pygame.mixer.init()
pygame.mixer.music.load(midi_file_name)
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    time.sleep(1)
