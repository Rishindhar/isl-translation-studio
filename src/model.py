import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.config import SEQUENCE_LENGTH, FEATURE_DIM, MODEL_PATH, LABELS_PATH, DATA_DIR

class ISL_GRU_Network(nn.Module):
    """
    Gated Recurrent Unit (GRU) Neural Network for ISL Recognition.
    Processes sequential 30-frame landmark matrices (30 x 63) using Bidirectional GRU
    layers with Dropout and linear classification layer.
    """
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.2):
        super(ISL_GRU_Network, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Bidirectional GRU Layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # Dense Classifier Output Layer (hidden_dim * 4 due to concat of mean and max pooling)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, feature_dim)
        gru_out, _ = self.gru(x)  # gru_out shape: (batch_size, sequence_length, hidden_dim * 2)
        
        # Mean + Max pooling across sequence dimension for high-fidelity temporal representation
        mean_pooled = torch.mean(gru_out, dim=1)
        max_pooled, _ = torch.max(gru_out, dim=1)
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

    def freeze_feature_extractor(self):
        """Freezes GRU layers for Transfer Learning fine-tuning."""
        for param in self.gru.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreezes all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def update_class_count(self, new_num_classes):
        """Expands dense output classification layer for dynamic vocabulary addition."""
        if new_num_classes == self.num_classes:
            return

        old_fc = self.fc
        in_features = old_fc.in_features
        new_fc = nn.Linear(in_features, new_num_classes)

        # Copy existing weights for existing classes
        with torch.no_grad():
            min_classes = min(self.num_classes, new_num_classes)
            new_fc.weight[:min_classes] = old_fc.weight[:min_classes]
            new_fc.bias[:min_classes] = old_fc.bias[:min_classes]

        self.fc = new_fc
        self.num_classes = new_num_classes


class ISLGestureClassifier:
    """Wrapper class managing model lifecycle, training, transfer learning, and live inference."""
    def __init__(self, labels_list=None, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        self.model_path = model_path
        self.labels_path = labels_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if labels_list is not None:
            self.labels = labels_list
        else:
            self.labels = self.load_labels() or ["hello", "thank_you", "yes", "no", "help", "idle"]

        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.idx_to_label = {i: lbl for i, lbl in enumerate(self.labels)}

        self.model = ISL_GRU_Network(
            input_dim=FEATURE_DIM,
            hidden_dim=64,
            num_layers=2,
            num_classes=len(self.labels)
        ).to(self.device)

        if os.path.exists(self.model_path):
            self.load_model(override_labels=labels_list is not None)

        if labels_list is not None:
            self.labels = labels_list
            self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
            self.idx_to_label = {i: lbl for i, lbl in enumerate(self.labels)}
            self.model.update_class_count(len(self.labels))

    def save_labels(self):
        os.makedirs(os.path.dirname(self.labels_path), exist_ok=True)
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)

    def load_labels(self):
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.save_labels()
        torch.save({
            "state_dict": self.model.state_dict(),
            "labels": self.labels,
            "num_classes": len(self.labels)
        }, self.model_path)

    def load_model(self, override_labels=False):
        if not os.path.exists(self.model_path):
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if not override_labels:
                saved_labels = checkpoint.get("labels", self.labels)
                self.labels = saved_labels
                self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
                self.idx_to_label = {i: lbl for i, lbl in enumerate(self.labels)}

            self.model.update_class_count(len(self.labels))
            if checkpoint["state_dict"]["fc.weight"].shape[0] == len(self.labels):
                self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            return True
        except Exception as e:
            print(f"Notice: Initializing fresh model weights due to architecture update ({e}).")
            return False

    def predict(self, sequence_array):
        """
        Runs GRU inference on a single 30x63 NumPy sequence matrix.

        :param sequence_array: NumPy array of shape (30, 63)
        :return: Tuple (predicted_label_str, confidence_float, proba_dict)
        """
        self.model.eval()
        if sequence_array.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            return "idle", 0.0, {}

        tensor_in = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        predicted_label = self.idx_to_label.get(best_idx, "idle")

        proba_dict = {self.idx_to_label[i]: float(probs[i]) for i in range(len(self.labels))}
        return predicted_label, confidence, proba_dict

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=30, batch_size=16, lr=0.001):
        """Trains the GRU network from scratch or full fine-tune."""
        self.model.train()
        self.model.unfreeze_all()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            acc = correct / max(total, 1)
            avg_loss = total_loss / max(total, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.1f}%", flush=True)

        self.save_model()

    def delete_custom_class(self, label):
        """
        Removes a custom class from the model and disk.
        Slices out the class row from the linear fc layer weights, updates self.labels, 
        deletes the files/folder from disk, and saves the updated checkpoint.
        """
        if label not in self.labels:
            print(f"[DELETE CLASS] Warning: Label '{label}' not found in model.")
            return False

        idx = self.label_to_idx[label]
        
        # 1. Update labels list
        self.labels.remove(label)
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.idx_to_label = {i: lbl for i, lbl in enumerate(self.labels)}

        # 2. Slice out weights from fc layer
        old_fc = self.model.fc
        in_features = old_fc.in_features
        new_num_classes = len(self.labels)
        
        new_fc = nn.Linear(in_features, new_num_classes)
        
        with torch.no_grad():
            # Slice weights and bias, removing row index `idx`
            weight_part1 = old_fc.weight[:idx]
            weight_part2 = old_fc.weight[idx+1:]
            new_fc.weight.copy_(torch.cat([weight_part1, weight_part2], dim=0))
            
            bias_part1 = old_fc.bias[:idx]
            bias_part2 = old_fc.bias[idx+1:]
            new_fc.bias.copy_(torch.cat([bias_part1, bias_part2], dim=0))

        self.model.fc = new_fc
        self.model.num_classes = new_num_classes

        # 3. Delete directory from disk
        import shutil
        sign_dir = os.path.join(DATA_DIR, label)
        if os.path.exists(sign_dir):
            shutil.rmtree(sign_dir)
            print(f"[DELETE CLASS] Deleted folder: {sign_dir}")

        # 4. Save model to disk
        self.save_model()
        print(f"[DELETE CLASS] Model and labels saved. Class '{label}' removed completely.")
        return True

    def get_custom_labels(self):
        """Discovers custom signs recorded on disk with live_webcam_*.npy files."""
        custom = []
        if os.path.exists(DATA_DIR):
            for folder in os.listdir(DATA_DIR):
                folder_path = os.path.join(DATA_DIR, folder)
                if os.path.isdir(folder_path):
                    # Check if there are live webcam files
                    if glob.glob(os.path.join(folder_path, "live_webcam_*.npy")):
                        custom.append(folder)
        return custom

    def fine_tune_transfer_learning(self, X_new, y_new, new_labels, epochs=15, lr=0.001):
        """
        Applies Transfer Learning:
        Freezes GRU backbone and fine-tunes updated linear classifier on newly added gesture data.
        Ensures baseline classes are preserved by restoring their original weights after training.
        """
        for lbl in new_labels:
            if lbl not in self.labels:
                self.labels.append(lbl)
 
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.labels)}
        self.idx_to_label = {i: lbl for i, lbl in enumerate(self.labels)}
 
        self.model.update_class_count(len(self.labels))
        self.model.to(self.device)

        # ── Weight Preservation ──────────────────────────────────────────────
        # Clone current fc weights and biases to preserve baseline classes
        original_fc_weight = self.model.fc.weight.clone().detach()
        original_fc_bias = self.model.fc.bias.clone().detach()

        # Find custom signs on disk to protect them from being overwritten by old weights
        custom_labels = self.get_custom_labels()
        for lbl in new_labels:
            if lbl not in custom_labels:
                custom_labels.append(lbl)

        # Identify baseline indexes (classes that are NOT custom signs)
        baseline_indexes = [self.label_to_idx[lbl] for lbl in self.labels if lbl not in custom_labels]
 
        # Freeze GRU backbone
        self.model.freeze_feature_extractor()
        self.model.train()
 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=lr)
 
        X_t = torch.tensor(X_new, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_new, dtype=torch.long).to(self.device)
 
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
 
        print(f"[TRANSFER LEARNING] Starting training for {epochs} epochs on {len(X_new)} samples...", flush=True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
                
                preds = torch.argmax(outputs, dim=1)
                correct += torch.sum(preds == batch_y).item()
                total += len(batch_x)
                
            avg_loss = epoch_loss / total
            accuracy = (correct / total) * 100
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%", flush=True)
 
        # ── Weight Restoration ───────────────────────────────────────────────
        # Restore pre-training weights and biases for all baseline classes
        print(f"[TRANSFER LEARNING] Restoring {len(baseline_indexes)} baseline class weights to prevent forgetting...", flush=True)
        with torch.no_grad():
            for idx in baseline_indexes:
                self.model.fc.weight[idx] = original_fc_weight[idx]
                self.model.fc.bias[idx] = original_fc_bias[idx]

        self.save_model()
        print("[TRANSFER LEARNING] Fine-tuning complete. Model saved.", flush=True)
