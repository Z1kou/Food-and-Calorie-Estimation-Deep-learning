import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers # type: ignore

def create_improved_classification_model(input_shape=(224, 224, 3), num_classes=19):
    """
    Creates an improved CNN model for food classification with:
    - Increased capacity
    - Better regularization
    - Residual connections
    - Proper initialization
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation layers
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomBrightness(0.2)(x)
    
    # Initial convolution
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Residual blocks
    def residual_block(x, filters, stride=1):
        shortcut = x
        
        # First convolution
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second convolution
        x = layers.Conv2D(filters, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride,
                                   padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    # Add residual blocks with increasing filters
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with stronger regularization
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='improved_food_classifier')
    
    return model

def get_training_config():
    """
    Returns improved training configuration
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
    )
    
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    
    return optimizer, loss

if __name__ == "__main__":
    model = create_improved_classification_model()
    model.summary()