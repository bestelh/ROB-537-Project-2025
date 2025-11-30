import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def data_input(npz_file_path="training_data_1.npz"):
    """Load training data from .npz file"""
    data = np.load(npz_file_path)
    x_train = data["x_train"]
    y_train = data["y_train"]
    return x_train, y_train


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class TransformerModel:
    def __init__(self, x_train, y_train, d_model=64, nhead=4, num_layers=2, device=None):

        self.x_train = x_train
        self.y_train = y_train

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.n_samples, self.input_dim = x_train.shape
        self.output_dim = y_train.shape[1]   # 576

        # ---------------------------
        # Model definition (Encoder only)
        # ---------------------------
        self.input_proj = nn.Linear(self.input_dim, d_model).to(self.device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True).to(self.device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)
        self.output_proj = nn.Linear(d_model, self.output_dim).to(self.device)

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.input_proj.parameters()) +
            list(self.output_proj.parameters()),
            lr=1e-3
        )

        self.train_losses = []


    # ======================================================
    # TRAINING LOOP
    # ======================================================

    def fit(self, epochs=10):
        x = torch.tensor(self.x_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
        for epoch in range(epochs):
            self.encoder.train()
            self.input_proj.train()
            self.output_proj.train()
            self.optimizer.zero_grad()
            src = self.input_proj(x).unsqueeze(1)  # (batch, seq=1, d_model)
            out = self.encoder(src)                # (batch, seq=1, d_model)
            pred = self.output_proj(out.squeeze(1))# (batch, 576)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
            self.train_losses.append(loss_val)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_val:.6f}")


    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, x_test):
        self.encoder.eval()
        self.input_proj.eval()
        self.output_proj.eval()
        x = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            src = self.input_proj(x).unsqueeze(1)
            out = self.encoder(src)
            pred = self.output_proj(out.squeeze(1))
        return pred.cpu().numpy()



# =====================================================================
# PLOTTING: LEARNING CURVE
# =====================================================================

def plot_learning_curve(loss_list):

    plt.figure(figsize=(7, 4))
    plt.plot(loss_list, label="Training Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()



# =====================================================================
# PLOTTING: TRUE VS PREDICTED FORCE CURVES
# =====================================================================

def plot_results(model, x_test, y_test, index=0):
    """
    Plots:
        - True and predicted Fx
        - True and predicted Fy
        - True and predicted Fz
    """

    pred = model.predict(x_test[index:index+1])[0]
    true = y_test[index]

    # reshape 576 → (3, 192)
    true_fx, true_fy, true_fz = np.split(true, 3)
    pred_fx, pred_fy, pred_fz = np.split(pred, 3)

    s = np.arange(192)

    plt.figure(figsize=(12, 8))

    # ---------------- FX ----------------
    plt.subplot(3, 1, 1)
    plt.plot(s, true_fx, label="True Fx")
    plt.plot(s, pred_fx, label="Pred Fx", linestyle="--")
    plt.title("Force-X Distribution")
    plt.legend()
    plt.grid()

    # ---------------- FY ----------------
    plt.subplot(3, 1, 2)
    plt.plot(s, true_fy, label="True Fy")
    plt.plot(s, pred_fy, label="Pred Fy", linestyle="--")
    plt.title("Force-Y Distribution")
    plt.legend()
    plt.grid()

    # ---------------- FZ ----------------
    plt.subplot(3, 1, 3)
    plt.plot(s, true_fz, label="True Fz")
    plt.plot(s, pred_fz, label="Pred Fz", linestyle="--")
    plt.title("Force-Z Distribution")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()



# =====================================================================
# MAIN
# =====================================================================

def main():

    print("Loading training data...")
    x_train, y_train = data_input("training_data_1.npz")

    print("Initializing transformer model...")
    model = TransformerModel(x_train, y_train)

    print("Training...")
    model.fit(epochs=25)

    print("Plotting learning curve...")
    plot_learning_curve(model.train_losses)

    print("Plotting example prediction...")
    plot_results(model, x_train, y_train, index=0)


# Run main if executed directly
if __name__ == "__main__":
    main()
