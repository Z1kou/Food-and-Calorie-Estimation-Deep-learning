import numpy as np
from Data.data_loader import ECUSTFDDataLoader
from Models import DeepFood
from Models.DeepFood import DeepFoodModel

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard # type: ignore
from datetime import datetime
import os


def create_loss_functions():
    """
    Création des fonctions de perte pour le modèle DeepFood
    """
    # Perte de classification
    classification_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0.1
    )
    
    # Perte de régression des boîtes
    regression_loss = tf.keras.losses.Huber(delta=1.0)
    
    # Perte RPN
    rpn_class_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=False
    )
    
    return classification_loss, regression_loss, rpn_class_loss

def train_deepfood_model(base_path, epochs=50, batch_size=16, initial_lr=1e-4):
    """
    Fonction d'entraînement du modèle DeepFood
    """
    # Chargement des données
    loader = ECUSTFDDataLoader(base_path)
    train_data, val_data, unique_labels = loader.prepare_data()
    
    # Création du modèle
    model = DeepFood(num_classes=len(unique_labels))
    
    # Définition des fonctions de perte
    classification_loss, regression_loss, rpn_class_loss = create_loss_functions()
    
    # Optimiseur
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    
    # Métriques
    train_classification_loss = tf.keras.metrics.Mean()
    train_regression_loss = tf.keras.metrics.Mean()
    train_rpn_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    @tf.function
    def train_step(images, labels, bboxes):
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(images, training=True)
            
            # Calcul des pertes
            cls_loss = classification_loss(labels, predictions['class_scores'])
            reg_loss = regression_loss(bboxes, predictions['bbox_deltas'])
            rpn_loss = rpn_class_loss(tf.ones_like(predictions['rpn_class']), 
                                    predictions['rpn_class'])
            
            # Perte totale
            total_loss = cls_loss + reg_loss + rpn_loss
        
        # Calcul des gradients et mise à jour
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Mise à jour des métriques
        train_classification_loss.update_state(cls_loss)
        train_regression_loss.update_state(reg_loss)
        train_rpn_loss.update_state(rpn_loss)
        train_accuracy.update_state(labels, predictions['class_scores'])
        
        return total_loss
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_deepfood_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Boucle d'entraînement
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Reset des métriques
        train_classification_loss.reset_states()
        train_regression_loss.reset_states()
        train_rpn_loss.reset_states()
        train_accuracy.reset_states()
        
        # Entraînement sur une époque
        for batch_images, batch_labels, batch_bboxes in create_batches(
            train_data['images'], 
            train_data['labels_onehot'],
            train_data.get('bboxes', None),  # Si disponible
            batch_size
        ):
            loss = train_step(batch_images, batch_labels, batch_bboxes)
            
        # Affichage des métriques
        print(f"Classification Loss: {train_classification_loss.result():.4f}")
        print(f"Regression Loss: {train_regression_loss.result():.4f}")
        print(f"RPN Loss: {train_rpn_loss.result():.4f}")
        print(f"Accuracy: {train_accuracy.result():.4f}")
        
    return model

def create_batches(images, labels, bboxes=None, batch_size=16):
    """
    Création des batchs pour l'entraînement
    """
    dataset = tf.data.Dataset.from_tensor_slices((
        images,
        labels,
        bboxes if bboxes is not None else tf.zeros_like(images[:, :, :, 0])
    ))
    
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset