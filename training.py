from    data_separation            import dataseparation
from    cnn                        import nn_sequencial
from    tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from    tensorflow.keras.models    import load_model
from    sklearn.metrics            import confusion_matrix, ConfusionMatrixDisplay
import  matplotlib.pyplot          as     plt
import  numpy                      as     np
import time
from datetime import datetime
start = datetime.now()
print(f"The process started at: {start}")

for i in range(5):
    x_val, x_test, x_train, y_val, y_test, y_train = dataseparation(target_size=(128,128),k=i+1)
    
    model = nn_sequencial(128)

    early_stopping = EarlyStopping(
        monitor              = 'val_loss',
        patience             = 5,
        mode                 = 'min',
        restore_best_weights = True
    )
    
    checkpoint = ModelCheckpoint(
        f'./models/ESP-K{i+1}.keras',
        monitor         = 'val_loss',
        save_best_only  = True,
        mode            = 'min'
    )

    history = model.fit(
        x_train, y_train,
        validation_data = (x_val,y_val),
        epochs          = 30,
        callbacks       = [early_stopping, checkpoint]
    )
    print("="*40)
    print(f"Results for k = {i+1}")
    loss, accuracy = model.evaluate(x_test, y_test)

    print("Accuracy:", f"{accuracy*100}%" )
    print("="*40)
    
    # --- PLOTTING TRAINING HISTORY FOR FOLD K ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Metrics - Fold {i+1}', fontsize=14, fontweight='bold')

    # 1. Plot Loss
    ax1.plot(history.history['loss'], label='Train Loss', color='blue', linestyle='--')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 2. Plot Accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linestyle='--')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"./metrics/History_K-{i+1}.png")

    # --- PLOTTING confusion_matrix FOR FOLD K ---
    # 1. Generate continuous raw probabilities from the model
    model   = load_model(f'./models-v1/ESP-K{i+1}.keras')
    pred    = model.predict(x_test)

    # 2. Convert probabilities to binary class labels (0 or 1)
    y_pred_classes = (pred > 0.5).astype(int)
    y_true_classes = np.array(y_test).astype(int)

    cm   = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Explive', 'Explosive'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d') 

    ax.set_title(f'Confusion Matrix - Fold {i+1}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)

    plt.savefig(f"./metrics/Confusion_Matrix_{i+1}.png", bbox_inches='tight')

    end = datetime.now()

print("="*80)
print(f"The process ended at: {end}")
print(f"The total time was: {end-start}")
print("="*80)
