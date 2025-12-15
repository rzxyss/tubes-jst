"""
Klasifikasi Buah Pepaya - Transfer Learning Model
Menggunakan MobileNetV2 Pre-trained
Tugas Akhir Jaringan Syaraf Tiruan

Author: [Nama Anda]
Dataset: Klasifikasi Pepaya (Mentah, Setengah Matang, Matang)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# ===========================
# 1. KONFIGURASI
# ===========================

class Config:
    # Path dataset
    TRAIN_DIR = 'output/train'
    VAL_DIR = 'output/val'
    TEST_DIR = 'output/test'
    
    # Hyperparameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    
    # Model settings
    NUM_CLASSES = 3  # mentah, setengah_matang, matang
    CLASS_NAMES = ['mature', 'partiallymature', 'unmature']
    BASE_MODEL = 'MobileNetV2'  # Options: 'MobileNetV2', 'EfficientNetB0', 'ResNet50'
    
    # Output
    MODEL_SAVE_PATH = f'models/pepaya_classifier_{BASE_MODEL}.h5'
    RESULTS_DIR = 'results_transfer_learning/'

config = Config()

# Buat folder jika belum ada
os.makedirs('models', exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

# ===========================
# 2. DATA PREPARATION
# ===========================

def create_data_generators():
    """
    Membuat data generators dengan augmentation yang lebih kuat
    """
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation dan Test hanya rescaling
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    # Load validation data
    val_generator = val_test_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Load test data
    test_generator = val_test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

# ===========================
# 3. MODEL ARCHITECTURE
# ===========================

def build_transfer_learning_model(base_model_name='MobileNetV2', trainable_layers=0):
    """
    Membuat model dengan transfer learning
    
    Args:
        base_model_name: Nama pre-trained model ('MobileNetV2', 'EfficientNetB0', 'ResNet50')
        trainable_layers: Jumlah layer terakhir yang akan di-fine-tune (0 = freeze all)
    """
    
    # Pilih base model
    if base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(
            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Jika ingin fine-tune beberapa layer terakhir
    if trainable_layers > 0:
        base_model.trainable = True
        # Freeze semua layer kecuali N layer terakhir
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    
    # Build model
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),
        
        # Base model (pre-trained)
        base_model,
        
        # Custom classification head
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(config.NUM_CLASSES, activation='softmax')
    ])
    
    return model, base_model

# ===========================
# 4. TWO-STAGE TRAINING
# ===========================

def two_stage_training(model, base_model, train_gen, val_gen):
    """
    Two-stage training:
    Stage 1: Train hanya classification head (base model frozen)
    Stage 2: Fine-tune beberapa layer terakhir base model
    """
    
    print("\n" + "="*60)
    print("STAGE 1: TRAINING CLASSIFICATION HEAD")
    print("="*60)
    
    # Stage 1: Train classification head
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks untuk stage 1
    callbacks_stage1 = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(
            f'models/pepaya_{config.BASE_MODEL}_stage1.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    history_stage1 = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=callbacks_stage1,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("STAGE 2: FINE-TUNING BASE MODEL")
    print("="*60)
    
    # Stage 2: Unfreeze beberapa layer terakhir untuk fine-tuning
    base_model.trainable = True
    
    # Freeze semua layer kecuali 30 layer terakhir
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"\nFine-tuning last {len(base_model.layers) - fine_tune_at} layers of base model")
    print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")
    
    # Compile dengan learning rate yang lebih kecil
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks untuk stage 2
    callbacks_stage2 = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1),
        ModelCheckpoint(
            config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    history_stage2 = model.fit(
        train_gen,
        epochs=70,
        validation_data=val_gen,
        callbacks=callbacks_stage2,
        verbose=1
    )
    
    # Debug: Print keys untuk memastikan
    print("\nHistory stage 1 keys:", history_stage1.history.keys())
    print("History stage 2 keys:", history_stage2.history.keys())
    
    # Combine histories dengan penanganan error yang lebih baik
    def get_history_value(history_dict, key, alternative_key=None):
        """Mendapatkan nilai dari history dengan fallback ke alternative key"""
        if key in history_dict:
            return history_dict[key]
        elif alternative_key and alternative_key in history_dict:
            return history_dict[alternative_key]
        else:
            # Cari key yang mengandung kata kunci
            for k in history_dict.keys():
                if key in k:
                    return history_dict[k]
            # Jika tidak ditemukan, return list kosong
            print(f"Warning: Key '{key}' tidak ditemukan dalam history")
            return []
    
    # Combine histories
    history = {
        'accuracy': history_stage1.history['accuracy'] + history_stage2.history['accuracy'],
        'val_accuracy': history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy'],
        'loss': history_stage1.history['loss'] + history_stage2.history['loss'],
        'val_loss': history_stage1.history['val_loss'] + history_stage2.history['val_loss'],
        'precision': get_history_value(history_stage1.history, 'precision', 'precision_1') + 
                     get_history_value(history_stage2.history, 'precision', 'precision_1'),
        'val_precision': get_history_value(history_stage1.history, 'val_precision', 'val_precision_1') + 
                         get_history_value(history_stage2.history, 'val_precision', 'val_precision_1'),
        'recall': get_history_value(history_stage1.history, 'recall', 'recall_1') + 
                  get_history_value(history_stage2.history, 'recall', 'recall_1'),
        'val_recall': get_history_value(history_stage1.history, 'val_recall', 'val_recall_1') + 
                      get_history_value(history_stage2.history, 'val_recall', 'val_recall_1'),
    }
    
    return history

# ===========================
# 5. EVALUATION FUNCTIONS
# ===========================

def plot_training_history(history):
    """
    Visualisasi training dan validation metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Accuracy
    axes[0, 0].plot(epochs, history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 1].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(epochs, history['precision'], 'b-', label='Train Precision', linewidth=2)
    axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='Val Precision', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(epochs, history['recall'], 'b-', label='Train Recall', linewidth=2)
    axes[1, 1].plot(epochs, history['val_recall'], 'r-', label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}training_history_{config.BASE_MODEL}.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history saved to {config.RESULTS_DIR}training_history_{config.BASE_MODEL}.png")
    plt.close()

def evaluate_model(model, test_gen):
    """
    Evaluasi model pada test set
    """
    print("\n" + "="*60)
    print("EVALUASI MODEL PADA TEST SET")
    print("="*60)
    
    # Predictions
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Classification Report
    print("\nCLASSIFICATION REPORT:")
    print("="*60)
    report = classification_report(y_true, y_pred, 
                                   target_names=config.CLASS_NAMES,
                                   digits=4)
    print(report)
    
    # Save report
    with open(f'{config.RESULTS_DIR}classification_report_{config.BASE_MODEL}.txt', 'w') as f:
        f.write(f"Model: {config.BASE_MODEL}\n")
        f.write("="*60 + "\n")
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {config.BASE_MODEL}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}confusion_matrix_{config.BASE_MODEL}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {config.RESULTS_DIR}confusion_matrix_{config.BASE_MODEL}.png")
    plt.close()
    
    # Calculate metrics
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen, verbose=0)
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    
    print("\n" + "="*60)
    print("FINAL TEST METRICS:")
    print("="*60)
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {f1_score:.4f}")
    print("="*60)
    
    return {
        'accuracy': test_acc,
        'loss': test_loss,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': f1_score,
        'confusion_matrix': cm
    }

def visualize_predictions(model, test_gen, num_images=16):
    """
    Visualisasi sample predictions
    """
    test_gen.reset()
    x_batch, y_batch = next(test_gen)
    predictions = model.predict(x_batch[:num_images], verbose=0)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(x_batch))):
        axes[i].imshow(x_batch[i])
        axes[i].axis('off')
        
        true_label = config.CLASS_NAMES[np.argmax(y_batch[i])]
        pred_label = config.CLASS_NAMES[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                         color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}sample_predictions_{config.BASE_MODEL}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Sample predictions saved to {config.RESULTS_DIR}sample_predictions_{config.BASE_MODEL}.png")
    plt.close()

# ===========================
# 6. MAIN EXECUTION
# ===========================

def main():
    """
    Main function untuk menjalankan seluruh pipeline
    """
    print("="*60)
    print(f"KLASIFIKASI BUAH PEPAYA - TRANSFER LEARNING")
    print(f"Model: {config.BASE_MODEL}")
    print("Tugas Akhir Jaringan Syaraf Tiruan")
    print("="*60)
    
    # Set random seed
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 1. Prepare Data
    print("\n[1/6] Loading dan preparing data...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    print(f"\nDataset Info:")
    print(f"  Training samples: {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Test samples: {test_gen.samples}")
    print(f"  Classes: {train_gen.class_indices}")
    
    # 2. Build Model
    print(f"\n[2/6] Building {config.BASE_MODEL} model...")
    model, base_model = build_transfer_learning_model(config.BASE_MODEL)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    
    # 3. Train Model (Two-stage)
    print("\n[3/6] Training model (two-stage)...")
    history = two_stage_training(model, base_model, train_gen, val_gen)
    
    # 4. Plot Training History
    print("\n[4/6] Plotting training history...")
    plot_training_history(history)
    
    # 5. Evaluate Model
    print("\n[5/6] Evaluating model...")
    results = evaluate_model(model, test_gen)
    
    # 6. Visualize Predictions
    print("\n[6/6] Visualizing sample predictions...")
    visualize_predictions(model, test_gen)
    
    # Save final results
    print("\n" + "="*60)
    print("TRAINING SELESAI!")
    print("="*60)
    print(f"\n✓ Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"✓ Results saved to: {config.RESULTS_DIR}")
    print(f"\nFinal Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Final Test F1-Score: {results['f1_score']:.4f}")
    
    return model, history, results

if __name__ == "__main__":
    # Cek GPU
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Run main pipeline
    model, history, results = main()