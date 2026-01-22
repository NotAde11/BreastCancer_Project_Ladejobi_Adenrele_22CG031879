"""
Neural Network Training Pipeline for Tumor Classification
----------------------------------------------------------
Implements a transfer learning approach using MobileNetV2 for
binary classification of breast tumor histopathology images.

Execution Guide for Google Colab:
1. Upload this script to your Colab notebook
2. Prepare image dataset following structure below
3. Execute all cells sequentially
4. Export the generated Model.h5 file

Expected Dataset Layout:
images/
├── training/
│   ├── healthy/        # Non-malignant tissue samples
│   └── cancerous/      # Malignant tissue samples
└── validation/
    ├── healthy/
    └── cancerous/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ------------------------------------
# Hyperparameters & Settings
# ------------------------------------

class TrainingConfig:
    """Centralized configuration for the training process."""
    
    # Input dimensions
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224
    COLOR_CHANNELS = 3
    
    # Training hyperparameters
    SAMPLES_PER_BATCH = 32
    MAX_ITERATIONS = 50
    INITIAL_LR = 0.0001
    
    # Data paths (adjust for your environment)
    TRAINING_IMAGES = 'dataset/train'
    VALIDATION_IMAGES = 'dataset/validation'
    
    # Output path
    OUTPUT_MODEL = 'Model.h5'
    
    # Classification labels
    LABEL_NAMES = ['benign', 'malignant']


# ------------------------------------
# Data Pipeline Construction
# ------------------------------------

def build_data_pipelines():
    """
    Constructs image data generators with augmentation for training
    and standard preprocessing for validation.
    
    Augmentation strategies help improve model generalization
    by introducing controlled variations in the training data.
    """
    
    # Training augmentation pipeline
    training_processor = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation preprocessing (normalization only)
    validation_processor = ImageDataGenerator(rescale=1.0/255.0)
    
    # Create training data generator
    training_pipeline = training_processor.flow_from_directory(
        TrainingConfig.TRAINING_IMAGES,
        target_size=(TrainingConfig.INPUT_HEIGHT, TrainingConfig.INPUT_WIDTH),
        batch_size=TrainingConfig.SAMPLES_PER_BATCH,
        class_mode='binary',
        shuffle=True
    )
    
    # Create validation data generator
    validation_pipeline = validation_processor.flow_from_directory(
        TrainingConfig.VALIDATION_IMAGES,
        target_size=(TrainingConfig.INPUT_HEIGHT, TrainingConfig.INPUT_WIDTH),
        batch_size=TrainingConfig.SAMPLES_PER_BATCH,
        class_mode='binary',
        shuffle=False
    )
    
    return training_pipeline, validation_pipeline


# ------------------------------------
# Network Architecture Definition
# ------------------------------------

def construct_classifier():
    """
    Builds the classification network using transfer learning.
    
    Leverages MobileNetV2 pretrained on ImageNet as feature extractor,
    adding custom dense layers for tumor classification.
    """
    
    # Initialize pretrained backbone
    backbone = MobileNetV2(
        input_shape=(TrainingConfig.INPUT_HEIGHT, TrainingConfig.INPUT_WIDTH, TrainingConfig.COLOR_CHANNELS),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze backbone weights initially
    backbone.trainable = False
    
    # Build complete architecture
    network = keras.Sequential([
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return network


# ------------------------------------
# Training Execution
# ------------------------------------

def execute_training(network, training_data, validation_data):
    """
    Runs the primary training loop with optimization callbacks.
    """
    
    # Configure optimizer and metrics
    network.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TrainingConfig.INITIAL_LR),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # Define training callbacks
    optimization_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7
        )
    ]
    
    # Execute training
    print("Initiating training sequence...")
    print(f"Training samples: {training_data.samples}")
    print(f"Validation samples: {validation_data.samples}")
    print(f"Batch size: {TrainingConfig.SAMPLES_PER_BATCH}")
    print(f"Max epochs: {TrainingConfig.MAX_ITERATIONS}\n")
    
    training_record = network.fit(
        training_data,
        epochs=TrainingConfig.MAX_ITERATIONS,
        validation_data=validation_data,
        callbacks=optimization_callbacks,
        verbose=1
    )
    
    return training_record


# ------------------------------------
# Fine-Tuning Phase
# ------------------------------------

def refine_network(network, training_data, validation_data):
    """
    Performs fine-tuning by unfreezing deeper backbone layers
    and continuing training with reduced learning rate.
    """
    
    print("\nInitiating fine-tuning phase...")
    
    # Unlock backbone layers
    feature_extractor = network.layers[0]
    feature_extractor.trainable = True
    
    # Keep early layers frozen
    for layer in feature_extractor.layers[:-20]:
        layer.trainable = False
    
    # Reconfigure with lower learning rate
    network.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TrainingConfig.INITIAL_LR / 10),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # Continue training
    refinement_record = network.fit(
        training_data,
        epochs=10,
        validation_data=validation_data,
        verbose=1
    )
    
    return refinement_record


# ------------------------------------
# Visualization Utilities
# ------------------------------------

def generate_training_charts(primary_record, refinement_record=None):
    """
    Creates visualization charts for training progression.
    """
    
    plt.figure(figsize=(15, 5))
    
    # Accuracy progression
    plt.subplot(1, 3, 1)
    plt.plot(primary_record.history['accuracy'], label='Train Accuracy')
    plt.plot(primary_record.history['val_accuracy'], label='Val Accuracy')
    if refinement_record:
        start_idx = len(primary_record.history['accuracy'])
        plt.plot(range(start_idx, start_idx + len(refinement_record.history['accuracy'])),
                refinement_record.history['accuracy'], label='Fine-tune Train')
        plt.plot(range(start_idx, start_idx + len(refinement_record.history['val_accuracy'])),
                refinement_record.history['val_accuracy'], label='Fine-tune Val')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss progression
    plt.subplot(1, 3, 2)
    plt.plot(primary_record.history['loss'], label='Train Loss')
    plt.plot(primary_record.history['val_loss'], label='Val Loss')
    if refinement_record:
        start_idx = len(primary_record.history['loss'])
        plt.plot(range(start_idx, start_idx + len(refinement_record.history['loss'])),
                refinement_record.history['loss'], label='Fine-tune Train')
        plt.plot(range(start_idx, start_idx + len(refinement_record.history['val_loss'])),
                refinement_record.history['val_loss'], label='Fine-tune Val')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # AUC progression
    plt.subplot(1, 3, 3)
    plt.plot(primary_record.history['auc'], label='Train AUC')
    plt.plot(primary_record.history['val_auc'], label='Val AUC')
    if refinement_record:
        start_idx = len(primary_record.history['auc'])
        plt.plot(range(start_idx, start_idx + len(refinement_record.history['auc'])),
                refinement_record.history['auc'], label='Fine-tune Train')
        plt.plot(range(start_idx, start_idx + len(refinement_record.history['val_auc'])),
                refinement_record.history['val_auc'], label='Fine-tune Val')
    plt.title('AUC Score Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


# ------------------------------------
# Performance Evaluation
# ------------------------------------

def assess_performance(network, validation_data):
    """
    Computes final performance metrics on validation set.
    """
    
    print("\n" + "=" * 50)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 50)
    
    metrics = network.evaluate(validation_data, verbose=1)
    
    print(f"\nValidation Loss: {metrics[0]:.4f}")
    print(f"Validation Accuracy: {metrics[1]:.4f}")
    print(f"Validation Precision: {metrics[2]:.4f}")
    print(f"Validation Recall: {metrics[3]:.4f}")
    print(f"Validation AUC: {metrics[4]:.4f}")


# ------------------------------------
# Main Pipeline Execution
# ------------------------------------

def run_pipeline():
    """
    Orchestrates the complete training workflow.
    """
    
    print("=" * 70)
    print("TUMOR CLASSIFICATION MODEL TRAINING")
    print("=" * 70)
    
    # Reproducibility setup
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Hardware detection
    print(f"\nGPU Devices: {tf.config.list_physical_devices('GPU')}")
    
    # Data pipeline setup
    print("\n[Phase 1/5] Constructing data pipelines...")
    train_data, val_data = build_data_pipelines()
    
    # Architecture construction
    print("\n[Phase 2/5] Building network architecture...")
    network = construct_classifier()
    network.summary()
    
    # Primary training
    print("\n[Phase 3/5] Executing primary training...")
    primary_record = execute_training(network, train_data, val_data)
    
    # Fine-tuning
    print("\n[Phase 4/5] Executing fine-tuning phase...")
    refinement_record = refine_network(network, train_data, val_data)
    
    # Performance assessment
    print("\n[Phase 5/5] Assessing final performance...")
    assess_performance(network, val_data)
    
    # Generate visualizations
    print("\nGenerating training visualizations...")
    generate_training_charts(primary_record, refinement_record)
    
    # Export model
    print(f"\nExporting model to {TrainingConfig.OUTPUT_MODEL}...")
    network.save(TrainingConfig.OUTPUT_MODEL)
    
    print("\n" + "=" * 70)
    print("[COMPLETE] Training pipeline finished successfully")
    print("=" * 70)
    print(f"\nExported model: {TrainingConfig.OUTPUT_MODEL}")
    print("Download and integrate with the Flask application.")
    print("\nNext steps:")
    print("1. Download Model.h5 from Colab")
    print("2. Place in project root directory")
    print("3. Launch the Flask application")


# ------------------------------------
# Dataset Setup Instructions
# ------------------------------------

def display_setup_guide():
    """
    Displays instructions for dataset preparation.
    
    Recommended datasets:
    - Breast Cancer Histopathology Images (Kaggle)
    - MIAS Mammography Database
    - INbreast Database
    """
    
    guide_text = """
    ================================================================
    DATASET CONFIGURATION GUIDE
    ================================================================
    
    1. Obtain a breast cancer image dataset. Suggested sources:
       - Kaggle: "Breast Histopathology Images"
       - https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
    
    2. Arrange files in this directory structure:
    
       dataset/
       ├── train/
       │   ├── benign/      (non-malignant samples)
       │   └── malignant/   (malignant samples)
       └── validation/
           ├── benign/      (validation non-malignant)
           └── malignant/   (validation malignant)
    
    3. Recommended data split:
       - 80% training set
       - 20% validation set
    
    4. Minimum sample requirements:
       - 500+ images per class (training)
       - 100+ images per class (validation)
    
    5. Upload to Colab or mount Google Drive
    
    6. Update TrainingConfig paths if necessary
    
    ================================================================
    """
    
    print(guide_text)


# Pipeline entry point
if __name__ == "__main__":
    # Uncomment below for setup instructions
    # display_setup_guide()
    
    run_pipeline()