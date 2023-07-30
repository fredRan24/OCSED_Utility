import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path 
import math
from keras.models import load_model

from data_preprocessing import Data_preprocessing
from training import Training_CRNN
from evaluation import Evaluator

class CustomEvaluatorCallback(tf.keras.callbacks.Callback):
        def __init__(self, eval_func):
            super().__init__()
            self.eval_func = eval_func

        def on_epoch_end(self, epoch, logs=None):
            print("\nRunning custom evaluation at the end of epoch", epoch)
            self.eval_func(self.model)
            print("\n")

class OCSED_util:
    def __init__(self):
        self.data_preprocessor = Data_preprocessing()
        self.trainer = Training_CRNN()
        self.evaluator = Evaluator()

        #config
        self.sr = 22050
        self.target_event_label = "Dog"

        #Spectrogram 
        self.hop_length = 512
        self.n_mels = 128
        self.n_ffts = 2048
        
        #File Info
        self.metadata = pd.DataFrame()
        self.threshold = 0.5
        self.max_duration = 10 #seconds
        self.max_time_steps = 431

        #Training Hyperparams
        self.n_epochs = 50
        self.learning_rate = 0.001
        self.batch_size = 16
        self.patience = 5

        #Testing Params
        self.collar = 0.200 #seconds

        #Paths
        self.current_dir = Path().parent.absolute()
        self.dataset_dir = self.current_dir / 'DESED' / 'dataset' 
        
        #Training Audio & Metadata Paths
        self.train_audio_dir = self.dataset_dir / 'audio' / 'train' / 'synthetic21_train' / 'soundscapes'
        self.train_metadata_path = self.dataset_dir / 'metadata' / 'train' / 'synthetic21_train' / 'soundscapes.tsv'
        
        #Testing Audio & Metadata Paths
        self.eval_audio_dir = self.dataset_dir / 'audio' / 'eval' / 'public'
        self.eval_metadata_path = self.dataset_dir / 'metadata' / 'eval' / 'public.tsv'

        #Validation Audio & Metadata Paths
        self.val_audio_dir = self.dataset_dir / 'audio'/ 'validation' / 'synthetic21_validation' / 'soundscapes'
        self.val_metadata_path = self.dataset_dir / 'metadata' / 'validation' / 'synthetic21_validation' / 'soundscapes.tsv'
        
        self.eval_f1_metric = CustomEvaluatorCallback(self.eval_model)

    # Functions for updating the config values
    def update_sr(self, sr):
        self.sr = sr
    
    def update_target_event_label(self, t_event_label):
        self.target_event_label = t_event_label

    def update_training_settings(self, n_epochs, learning_rate, batch_size, patience):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience

    def update_threshold(self, threshold):
        self.threshold = threshold

    def update_spctorgram_settings(self, hop_length, n_mels, n_ffts):
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_ffts = n_ffts

    def update_max_duration_and_tsteps(self, new_max_duration):
        self.max_duration = new_max_duration
        self.max_time_steps = math.ceil((self.sr * self.max_duration) / self.hop_length)

    def update_paths(self, dataset_dir, a_train, m_train, a_eval, m_eval, a_val, m_val):
        self.dataset_dir = dataset_dir 
        
        #Update Training Audio & Metadata Paths
        self.train_audio_dir = a_train
        self.train_metadata_path = m_train
        
        #Update Testing Audio & Metadata Paths
        self.eval_audio_dir = a_eval
        self.eval_metadata_path = m_eval

        #Update Validation Audio & Metadata Paths
        self.val_audio_dir = a_val
        self.val_metadata_path = m_val

    def update_test_settings(self, collar):
        self.collar = collar
        
    def preprocess_data(self): 
        data = self.data_preprocessor.prepare_combined_data(meta_a=self.train_metadata_path, 
                                                            audio_a=self.train_audio_dir, 
                                                            meta_b=self.val_metadata_path, 
                                                            audio_b=self.val_audio_dir, 
                                                            max_ts=self.max_time_steps, 
                                                            hop_len=self.hop_length, 
                                                            sr=self.sr, n_ffts=self.n_ffts, 
                                                            n_mels=self.n_mels)
        
        X_train, X_val, y_train, y_val = self.data_preprocessor.split_data(data)
        return X_train, X_val, y_train, y_val
    
    def train_model(self, model, X_train, X_val, y_train, y_val):
        # Get the length of the labels
        label_length = y_train.shape[1]
        print('Label length: ', label_length)

        #Get the model ready for training
        input_shape = (self.n_mels, self.max_time_steps, 1)
        model = self.trainer.create_crnn(input_shape)
        train_history = self.trainer.train_model(model, 
                                                 X_train, 
                                                 y_train, 
                                                 X_val, 
                                                 y_val, 
                                                 epochs=self.n_epochs, 
                                                 batch_size=self.batch_size,
                                                 patience=self.patience,
                                                 eval_metric=self.eval_f1_metric)
        
        return model, train_history
    
    def get_eval_f1_metric(self):
        return self.eval_f1_metric
    
    def eval_model(self, model):
        test_data = self.data_preprocessor.prepare_data(md=self.eval_metadata_path, 
                                                        ad=self.eval_audio_dir, 
                                                        max_ts=self.max_time_steps, 
                                                        hop_length=self.hop_length, 
                                                        sr=self.sr, n_ffts=self.n_ffts, 
                                                        n_mels=self.n_mels, 
                                                        include_non_dog_events=True)

        features, labels = zip(*test_data)

        # Convert features and labels to NumPy arrays
        features = np.array(features)
        labels = np.array(labels)

        y_eval = labels
        X_eval = features

        self.evaluator.evaluate(model=model, 
                                threshold=self.threshold, 
                                X_eval=X_eval, 
                                y_eval=y_eval, 
                                sr=self.sr, 
                                hop_length=self.hop_length, 
                                event_label=self.target_event_label, 
                                collar=self.collar)
        
    def save_model(self, model, save_directory, model_name):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
    
        model_path = os.path.join(save_directory, f"{model_name}.h5")
        if not os.path.exists(model_path):
            model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("Error, there was already a model saved at the specified location. \nPlease change location and try again...")

    def load_trained_model(self, model_directory, model_name):
        # Check if the model and weights files exist
        model_path = os.path.join(model_directory, f"{model_name}.h5")

        if not os.path.exists(model_path):
            print("Error: Model or weights file not found.")
            return None

        # Load the model
        model = load_model(model_path)

        print(f"Model loaded from {model_path}")

        # Compile the model (you can customize the optimizer, loss, and metrics according to your requirements)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

        return model
    
   
