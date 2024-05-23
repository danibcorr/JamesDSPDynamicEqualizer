# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------


import tensorflow as tf
from tensorflow import keras
import numpy as np


# --------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# --------------------------------------------------------------------------------------------


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):

    """
    Custom learning rate schedule that combines warm-up and cosine annealing.
    """

    def __init__(self, lr_start, lr_max, warmup_steps, total_steps):

        """
        Initializes the WarmUpCosine learning rate schedule.

        Args:
            lr_start (float): Initial learning rate.
            lr_max (float): Maximum learning rate.
            warmup_steps (int): Number of warm-up steps.
            total_steps (int): Total number of steps.
        """

        super(WarmUpCosine, self).__init__()

        self.lr_start = lr_start
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.pi = tf.constant(np.pi)


    def __call__(self, step):

        """
        Computes the learning rate for a given step.

        Args:
            step (tf.Tensor): Current step.

        Returns:
            tf.Tensor: Learning rate.
        """

        if self.total_steps < self.warmup_steps:

            raise ValueError(
                f"Total number of steps {self.total_steps} must be"
                + f" larger or equal to warmup steps {self.warmup_steps}."
            )

        # Compute cosine annealed learning rate
        cos_annealed_lr = tf.cos(self.pi * (tf.cast(step, tf.float32) - self.warmup_steps) / tf.cast(self.total_steps - self.warmup_steps, tf.float32))
        learning_rate = 0.5 * self.lr_max * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:

            if self.lr_max < self.lr_start:

                raise ValueError(
                    f"lr_start {self.lr_start} must be smaller or"
                    + f" equal to lr_max {self.lr_max}."
                )

            # Compute warm-up learning rate
            slope = (self.lr_max - self.lr_start) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.lr_start

            # Adjust learning rate based on warm-up steps
            learning_rate = tf.where(step < self.warmup_steps, warmup_rate, learning_rate)

        # Clip learning rate to 0.0 if step exceeds total_steps
        return tf.where(step > self.total_steps, 0.0, learning_rate, name="learning_rate")


    def get_config(self):

        """
        Gets configuration parameters of the learning rate schedule.

        Returns:
            dict: Configuration parameters.
        """

        config = {
            "lr_start": self.lr_start,
            "lr_max": self.lr_max,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }
        
        return config