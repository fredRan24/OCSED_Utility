import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

class Data_preprocessing():
    def __init__(self):
        self.name = "Alfie's Preprocessor"

    def read_annotations(self, file_path):
        metadata = pd.read_table(file_path)

        # Filter the metadata to only include dog barks
        dog_barks = metadata[metadata['event_label'] == 'Dog']

        # Group the dog barks by filename
        dog_bark_grouped = dog_barks.groupby('filename')

        # Filter the metadata to only include non-dog events
        non_dog_events = metadata[metadata['event_label'] != 'Dog']

        # Get the unique filenames for non-dog events
        non_dog_filenames = non_dog_events['filename'].unique()

        # Get the number of dog_files
        n_dog_files = int(len(dog_bark_grouped) / 3)

        # Randomly select n_dog_files from the non-dog event files
        non_dog_filenames_sampled = np.random.choice(non_dog_filenames, size=n_dog_files, replace=False)

        # Create a DataFrame with the selected non-dog event files
        non_dog_grouped = non_dog_events[non_dog_events['filename'].isin(non_dog_filenames_sampled)].groupby('filename')

        return dog_bark_grouped, non_dog_grouped

    def create_log_mel_spectrogram(self, audio_path, sr, n_fft, hop_length, n_mels, target_frames=431):
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        # Pad or truncate the log-mel spectrogram along the time axis
        if log_mel_spectrogram.shape[1] < target_frames:
            padding = target_frames - log_mel_spectrogram.shape[1]
            log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, padding)))
        elif log_mel_spectrogram.shape[1] > target_frames:
            log_mel_spectrogram = log_mel_spectrogram[:, :target_frames]

        return log_mel_spectrogram

    def create_binary_matrix(self, group, time_steps, hop_length, sr):
        binary_matrix = np.zeros((time_steps, 1), dtype=int)

        for _, row in group.iterrows():
            onset_frame = librosa.time_to_frames(row['onset'], sr=sr, hop_length=hop_length)
            offset_frame = librosa.time_to_frames(row['offset'], sr=sr, hop_length=hop_length)
            binary_matrix[onset_frame:offset_frame] = 1
        ''' For DEBUGGING
        values = []
        for entry in binary_matrix:
            values.append(entry[0])
        print("Created matrix with: ", values, end="\r")
        '''
        return binary_matrix
    
    def pad_log_mel_spectrogram(self, log_mel_spec, max_time_steps):
        padding = max_time_steps - log_mel_spec.shape[1]
        if padding > 0:
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, padding)))
        return log_mel_spec

    def prepare_data(self, md, ad, max_ts, hop_length, sr, n_ffts, n_mels, include_non_dog_events=True):
        dog_bark_events, non_dog_events = self.read_annotations(md)
        data = []

        # Create a dictionary to store binary matrices for each file
        binary_matrices = {}

        for filename, group in dog_bark_events:
            # Create a binary matrix for the dog bark events in the current file
            binary_matrix = self.create_binary_matrix(group, max_ts, hop_length, sr=sr)

            # If the binary matrix for this file already exists, update it by combining the matrices
            if filename in binary_matrices:
                binary_matrices[filename] = np.maximum(binary_matrices[filename], binary_matrix)
            else:
                binary_matrices[filename] = binary_matrix
        
        #Pass False, if you just want to get dog event
        if include_non_dog_events==True:
            for filename in non_dog_events.groups:
                binary_matrices[filename] = np.zeros((431,1))
        
        num_files = len(binary_matrices)
        print('Number of files to convert to spectrograms: ', num_files)
        for index, (filename, binary_matrix) in enumerate(binary_matrices.items()):
            path = os.path.join(ad, filename)
            log_mel_spec = self.create_log_mel_spectrogram(path, sr, n_fft=n_ffts, hop_length=hop_length, n_mels=n_mels, target_frames=max_ts)
            log_mel_spec = self.pad_log_mel_spectrogram(log_mel_spec, max_ts)

            data.append((log_mel_spec, binary_matrix))

            # Print the progress percentage
            progress = (index + 1) / num_files * 100
            print(f"Spectrogram Calc Progress: {progress:.2f}%", end="\r")

        return data

    #Prepares a combination of two sets of data
    def prepare_combined_data(self, meta_a, audio_a, meta_b, audio_b, max_ts, hop_len, sr, n_ffts, n_mels):
        synth_data = self.prepare_data(meta_a, audio_a, max_ts, hop_len, sr, n_ffts, n_mels)
        real_data = self.prepare_data(meta_b, audio_b, max_ts, hop_len, sr, n_ffts, n_mels)
    
        data = real_data + synth_data
        return data

    #Splits the data using sklearn train_test_split
    def split_data(self, data, test_size=0.1, random_state=42):
        f, l = zip(*data)

        # Convert features and labels to NumPy arrays
        features = np.array(f)
        labels = np.array(l)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=test_size, random_state=random_state)

        return X_train, X_val, y_train, y_val
