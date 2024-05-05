import torch
import torch.nn as nn
import numpy as np
from mido import Message, MidiFile, MidiTrack



class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MusicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

def generate_random_music(batch_size, sequence_length, input_size):
    return torch.randint(0, 2, (batch_size, sequence_length, input_size)).float()

def save_midi(music, filename='output.mid'):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for note in music:
        note = int(note)  # Convert from tensor if necessary
        # Assuming a velocity of 64 and a duration of 1 second (480 ticks)
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=480))

    mid.save(filename)
# Parameters
input_size = 88  # Number of piano keys
hidden_size = 256  # Number of LSTM units
output_size = 88  # Output size same as input (one hot encoded MIDI notes)

# Initialize model
model = MusicLSTM(input_size, hidden_size, output_size)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 50
batch_size = 32
sequence_length = 50  # Length of music sequences

for epoch in range(num_epochs):
    model.train()
    hidden = None  # Reset hidden state at the start of each sequence
    batch = generate_random_music(batch_size, sequence_length, input_size)
    target = batch  # Using the same data as target for simplicity
    optimizer.zero_grad()
    output, hidden = model(batch, hidden)
    loss = loss_function(output.transpose(1, 2), target.argmax(dim=2))  # Cross-entropy expects class indices
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            start = generate_random_music(1, 1, input_size)
            music, _ = model(start, None)
            for _ in range(sequence_length - 1):
                next_note, _ = model(music[:, -1:, :])
                music = torch.cat((music, next_note), dim=1)
            music = torch.argmax(music, dim=2)
            print("Generated Music (Piano Key Indices):", music.numpy())
            save_midi(music[0], filename=f'generated_music_epoch_{epoch+1}.mid')
