import numpy as np
from Data.data_loader import ECUSTFDDataLoader
from Models.cnn_model import create_improved_classification_model
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard # type: ignore
from datetime import datetime
import os

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        tf.summary.scalar('learning_rate', data=lr, step=epoch)

def create_data_augmentation():
    """
    Creates a more robust data augmentation pipeline
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])

def create_train_dataset(images, labels, batch_size, augmentation):
    """
    Creates an optimized tf.data.Dataset with augmentation
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_val_dataset(images, labels, batch_size):
    """
    Creates an optimized tf.data.Dataset for validation
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_classification_model(base_path, epochs=50, batch_size=32, initial_lr=1e-3):
    """
    Enhanced training function with better practices and monitoring
    """
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join('training_outputs', timestamp)
    model_dir = os.path.join(output_dir, 'models')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Data preparation
    loader = ECUSTFDDataLoader(base_path)
    train_data, val_data, unique_labels = loader.prepare_data()

    # Print dataset information
    print(f"Dataset Summary:")
    print(f"Number of training samples: {len(train_data['images'])}")
    print(f"Number of validation samples: {len(val_data['images'])}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Image shape: {train_data['images'].shape[1:]}")

    # Create model
    model = create_improved_classification_model(
        input_shape=(224, 224, 3),
        num_classes=len(unique_labels)
    )

    # Create data augmentation
    augmentation = create_data_augmentation()

    # Create optimized datasets
    train_dataset = create_train_dataset(
        train_data['images'],
        train_data['labels_onehot'],
        batch_size,
        augmentation
    )
    val_dataset = create_val_dataset(
        val_data['images'],
        val_data['labels_onehot'],
        batch_size
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(model_dir, 'latest_model.weights.h5'),
            save_weights_only=True,
            verbose=0
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        ),
        LearningRateLogger()
    ]

    # Compile model with improved settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model and labels
    model.save(os.path.join(model_dir, 'final_model.keras'))
    np.save(os.path.join(output_dir, 'unique_labels.npy'), unique_labels)

    # Plot and save training history
    plot_training_history(history, output_dir)
    
    return model, history

def plot_training_history(history, output_dir):
    """
    Enhanced plotting function with more metrics
    """
    metrics = ['accuracy', 'top_3_accuracy', 'auc', 'loss']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Model {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

if __name__ == "__main__":
    base_path = r"C:\Users\salah\OneDrive\Bureau\Food Classification Project\Data\ECUSTFD-resized-"
    model, history = train_classification_model(base_path)