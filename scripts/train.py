from utils import neural_networks, preprocessing
from numpy.typing import NDArray
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score
from config import PROJECT_ROOT, BATCH_SIZE, DEVICE, EPOCHS


def train_model(model : dict, train_loader : DataLoader, dev_loader : DataLoader, model_name: str) -> tuple[float, dict]:
    model["model"].to(DEVICE)
    best_dev_f1 = 0.0
    best_model_state = None

    for epoch in range(EPOCHS):
        model["model"].train()
        running_loss = 0.0
        train_predictions = list()
        train_labels = list()
        
        for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                model["optimizer"].zero_grad()
                outputs = model["model"](images)
                loss = model["loss_fn"](outputs, labels)
                loss.backward()
                model["optimizer"].step()

                running_loss += loss.item() * images.size(0)
                
                # Collect training predictions
                _, predicted = torch.max(outputs, 1)
                train_predictions.append(predicted.cpu())
                train_labels.append(labels.cpu())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Calculate training metrics
        train_predictions = torch.cat(train_predictions)
        train_labels = torch.cat(train_labels)
        train_accuracy = (train_predictions == train_labels).sum().item() / len(train_labels)
        train_f1 = f1_score(train_labels, train_predictions, average="weighted")

        # Evaluate on dev set
        model["model"].eval()
        all_predictions = list()
        all_labels = list()
        with torch.no_grad():
            for images, labels in dev_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model["model"](images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.append(predicted.cpu())
                all_labels.append(labels.cpu())

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        dev_accuracy = (all_predictions == all_labels).sum().item() / len(all_labels)
        dev_f1 = f1_score(all_labels, all_predictions, average="weighted")
        
        # Track best dev F1 score and save best model state
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_model_state = model["model"].state_dict().copy()

        print(f"{model_name} - Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | "
              f"Train Acc: {train_accuracy:.4f}, Train F1: {train_f1:.4f} | "
              f"Dev Acc: {dev_accuracy:.4f}, Dev F1: {dev_f1:.4f}")
    
    return best_dev_f1, best_model_state


def main():
    data_dir = PROJECT_ROOT / "data" / "MNIST" / "raw"
    
    # Load MNIST as numpy array
    full_train_images : NDArray[np.uint8] = preprocessing.load_mnist_images(str(data_dir / "train-images-idx3-ubyte"))
    full_train_labels : NDArray[np.uint8] = preprocessing.load_mnist_labels(str(data_dir / "train-labels-idx1-ubyte"))
    full_train_images = preprocessing.preprocess(full_train_images)

    test_images : NDArray[np.uint8] = preprocessing.load_mnist_images(str(data_dir / "t10k-images-idx3-ubyte"))
    test_labels : NDArray[np.uint8] = preprocessing.load_mnist_labels(str(data_dir / "t10k-labels-idx1-ubyte"))
    test_images = preprocessing.preprocess(test_images)

    #Convert to Tensors
    full_train_images_tensor = torch.tensor(full_train_images, dtype=torch.float32)
    full_train_labels_tensor = torch.tensor(full_train_labels, dtype=torch.long)

    test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    #Convert to dataset
    full_train_tensor_dataset = TensorDataset(full_train_images_tensor, full_train_labels_tensor)
    test_images_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    #split ratio
    train_set_size = int(0.8 * len(full_train_images))
    dev_set_size = len(full_train_images) - train_set_size

    #split dataset
    train_set, dev_set = random_split(full_train_tensor_dataset, [train_set_size, dev_set_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_images_dataset, batch_size=BATCH_SIZE, shuffle=False)

    models  : list[dict] = neural_networks.retrieve_models()

    best_overall_model_idx = 0
    best_overall_f1 = 0.0
    best_overall_model_state = None

    for idx, model in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"Training Model {idx}")
        print(f"{'='*80}")
        best_f1, best_state = train_model(model, train_loader, dev_loader, f"Model {idx}")
        
        print(f"Best Dev F1 for Model {idx}: {best_f1:.4f}")
        
        # Track best model across all models
        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_overall_model_idx = idx
            best_overall_model_state = best_state
    
    # Save best model
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "best_model.pth"
    torch.save(best_overall_model_state, best_model_path)
    
    print(f"\n{'='*80}")
    print(f"Best Model: Model {best_overall_model_idx} with Dev F1: {best_overall_f1:.4f}")
    print(f"Model saved to: {best_model_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()