import os
import tensorflow as tf


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_accuracy', mode='max'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = None
        self.latest_checkpoint = None

        # Ensure the directory exists (if any directory is part of the path)
        directory = os.path.dirname(filepath)
        if directory:  # Only create if directory is specified
            os.makedirs(directory, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Get the monitored metric
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # Determine whether the metric improved
        if self.best is None or (self.mode == 'max' and current > self.best) or (self.mode == 'min' and current < self.best):
            self.best = current
            
            # Format the filepath with epoch and metric values
            filename = self.filepath.format(epoch=epoch+1, **logs)
            
            # Save the new model
            self.model.save(filename)
            
            # Delete the previous checkpoint after saving the new one
            if self.latest_checkpoint and os.path.exists(self.latest_checkpoint):
                os.remove(self.latest_checkpoint)
            
            # Update the latest checkpoint
            self.latest_checkpoint = filename