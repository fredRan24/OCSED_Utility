import librosa
import numpy as np
import dcase_util
import sed_eval

class Evaluator():
    # Extract events from a binary matrix
    def extract_events(self, binary_matrix, leniency=5, min_event_length=10):
        events = []
        onset, offset = None, None
        counter = 0

        # Iterate through the binary matrix
        for i, val in enumerate(binary_matrix):
            if val == 1:
                # If the value is 1 and onset is not set, set the onset to the current index
                if onset is None:
                    onset = i
                counter = 0
            elif val == 0:
                # If the value is 0 and onset is set, increment the counter
                if onset is not None:
                    counter += 1

                    # If the counter is greater than the leniency, set the offset and calculate event length
                    if counter > leniency:
                        offset = i - counter
                        event_length = offset - onset + 1
                        # If the event length is greater than the minimum event length, append the event to the list
                        if event_length > min_event_length:
                            events.append((onset, offset))
                        # Reset the onset and offset
                        onset, offset = None, None

        # If the onset is set and offset is not, calculate the event length and append the event if it's long enough
        if onset is not None and offset is None:
            offset = len(binary_matrix) - 1
            event_length = offset - onset + 1
            if event_length > min_event_length:
                events.append((onset, offset))

        return events

    # Convert event frames to time (in seconds)
    def frames_to_time(self, events, sr, hop_length):
        time_events = []

        # Convert each event's onset and offset frames to time using librosa
        for onset, offset in events:
            onset_time = librosa.frames_to_time(onset, sr=sr, hop_length=hop_length)
            offset_time = librosa.frames_to_time(offset, sr=sr, hop_length=hop_length)
            time_events.append((onset_time, offset_time))

        return time_events

    # Get the ground truth and prediction events in time format
    def get_ground_truth_and_prediction_t_events(self, ground_truth, prediction, sr, hop_length):
        ground_truth_events = self.extract_events(ground_truth)
        prediction_events = self.extract_events(prediction)

        ground_truth_time_events = self.frames_to_time(ground_truth_events, sr, hop_length)
        prediction_time_events = self.frames_to_time(prediction_events, sr, hop_length)

        return ground_truth_time_events, prediction_time_events

    # Get the ground truth and prediction binary matrices
    def get_ground_truth_and_prediction(self, model, index, threshold, X_val, y_val):
        truth_binary = y_val[index]
        log_mel_spectrogram = X_val[index]
        log_mel_spectrogram_expanded = np.expand_dims(log_mel_spectrogram, axis=0)
        
        #Get the prediction using the model
        prediction = model.predict(log_mel_spectrogram_expanded)

        #Flatten the predictions
        flattened_pred = prediction.flatten()
        
        # Threshold the prediction values to get binary output
        prediction_binary = (flattened_pred > threshold).astype(int)

        #Convert the values into lists
        truth = []
        for entry in truth_binary:
            truth.append(entry[0])

        pred = []
        for entry in prediction_binary:
            pred.append(entry)

        return truth, pred

    def generate_reference_and_estimated_files(self, model, threshold, X_val, y_val, sr, hop_length, event_label):
        reference_list = []
        estimated_list = []

        for index in range(len(X_val)):
            ground_truth, prediction = self.get_ground_truth_and_prediction(model, index, threshold, X_val, y_val)

            # Print onset and offset times
            ground_truth_events, predicted_events = self.get_ground_truth_and_prediction_t_events(ground_truth, prediction, sr, hop_length)

            # The events are already in time format, so there's no need for additional conversion
            ground_truth_time_events = ground_truth_events
            prediction_time_events = predicted_events

            for ground_truth_event in ground_truth_time_events:
                reference_list.append(dcase_util.containers.MetaDataItem({
                    'event_label': event_label,
                    'event_onset': ground_truth_event[0],
                    'event_offset': ground_truth_event[1],
                    'file': f'audio/{event_label}/reference_{index}.wav',
                    'scene_label': event_label
                }))

            for prediction_event in prediction_time_events:
                estimated_list.append(dcase_util.containers.MetaDataItem({
                    'event_label': event_label,
                    'event_onset': prediction_event[0],
                    'event_offset': prediction_event[1],
                    'file': f'audio/{event_label}/estimated_{index}.wav',
                    'scene_label': event_label
                }))

        return reference_list, estimated_list
    
    def binary_matrix_to_event_list(self, binary_matrix, sr, hop_length, event_names):
        events = []
        for i, row in enumerate(binary_matrix):
            onset, offset = None, None
            for j, val in enumerate(row):
                if val == 1:
                    if onset is None:
                        onset = j * hop_length / sr
                else:
                    if onset is not None:
                        offset = j * hop_length / sr
                        events.append({'event_label': event_names[i], 'onset': onset, 'offset': offset})
                        onset, offset = None, None
        return events

    def get_predictions(self, model, features):
        # Reshape features to have an extra dimension for the batch size
        features_reshaped = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)

        # Get the predictions
        predictions = model.predict(features_reshaped)

        # Reshape predictions to match the original labels shape
        predictions_reshaped = predictions.reshape(predictions.shape[0], predictions.shape[1], 1)

        return predictions_reshaped
    
    def evaluate(self, model, threshold, X_eval, y_eval, sr, hop_length, event_label, collar):
        reference_data, estimated_data = self.generate_reference_and_estimated_files(model, threshold, X_eval, y_eval, sr, hop_length, event_label)
        print("#Ref Events: ", len(reference_data), "\n#Est Events: ", len(estimated_data))
        #Get f1 score from data
        '''
        for index, (ref_event, est_event) in enumerate(zip(reference_data, estimated_data)):
            print(f"File {index}:")
            print("Reference events:")
            print(ref_event)
            print("Estimated events:")
            print(est_event)
            print("\n")
        '''
        data = []

        for index in range(len(X_eval)):
            reference_events = [ref for ref in reference_data if ref['file'] == f'audio/{event_label}/reference_{index}.wav']
            estimated_events = [est for est in estimated_data if est['file'] == f'audio/{event_label}/estimated_{index}.wav']

            reference_event_list = dcase_util.containers.MetaDataContainer(reference_events)
            estimated_event_list = dcase_util.containers.MetaDataContainer(estimated_events)

            data.append({
                'reference_event_list': reference_event_list,
                'estimated_event_list': estimated_event_list
            })

        all_data = dcase_util.containers.MetaDataContainer()
        for file_pair in data:
            all_data += file_pair['reference_event_list']

        event_labels = all_data.unique_event_labels

        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=event_labels,
            t_collar=collar
        )

        for file_pair in data:
            event_based_metrics.evaluate(
                reference_event_list=file_pair['reference_event_list'],
                estimated_event_list=file_pair['estimated_event_list']
            )

        print("CW F1:", event_based_metrics.results_overall_metrics()) #As this is a one-class system this is ok to do