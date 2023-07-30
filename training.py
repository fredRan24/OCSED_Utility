import numpy as np
import tensorflow as tf

class Training_CRNN():
    def train_model(self, model, X_train, y_train, X_val, y_val, epochs, batch_size, patience, eval_metric):
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
        y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
        
        evaluation_EB_F1 = eval_metric

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
                [evaluation_EB_F1]
            ]
        )
        return history
    
    