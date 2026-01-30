# 🏠 House Price Prediction with PyTorch Deep Learning
## MLP (Multi-Layer Perceptron) Neural Network
## 🎯 Project Overview

This project implements a **Multi-Layer Perceptron (MLP)** using PyTorch to predict house prices based on 80+ features.

**Key Highlights:**
- 📊 Deep Learning with PyTorch
- 🧠 4-layer Neural Network architecture
- 📈 RMSE optimization
- 🎓 Week 3 AI Engineering Assignment

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Kaggle RMSE** | **0,23174** |
| Validation RMSE | $37,101.24 |
| Train RMSE | $27,665.06 |
| Overfitting Ratio | 34.11% |

---

## 🧠 Model Architecture
```
Input Layer (80 features)
    ↓
Linear(80 → 128) + ReLU + Dropout(0.2)
    ↓
Linear(128 → 64) + ReLU + Dropout(0.2)
    ↓
Linear(64 → 32) + ReLU
    ↓
Linear(32 → 1)
    ↓
Output (SalePrice)
```

**Total Parameters:** ~20,737

---

## 🔧 Tech Stack

- **PyTorch** - Deep Learning framework
- **Python 3.8+**
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Preprocessing (StandardScaler, LabelEncoder)
- **Matplotlib & Seaborn** - Visualization
- **Google Colab** - Development environment

---

## 📁 Project Structure
```
house-prices-pytorch-deep-learning/
│
├── notebooks/           # Jupyter notebook with full pipeline
├── models/              # Trained PyTorch model (.pth)
├── submissions/         # Kaggle submission file
├── screenshots/         # Training visualizations
├── data/                # Dataset (or Kaggle link)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Notebook

1. Clone this repository:
```bash
git clone https://github.com/[username]/house-prices-pytorch-deep-learning.git
cd house-prices-pytorch-deep-learning
```

2. Open the notebook:
```bash
jupyter notebook notebooks/House_Prices_PyTorch_MLP.ipynb
```

3. Run all cells

---

## 📈 Training Process

### Loss Curves

![Training Loss](screenshots/02_training_loss.png)

**Training Details:**
- **Epochs:** 100
- **Batch Size:** 32
- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** MSELoss

**Observations:**
- Both train and validation losses decreased steadily
- Minimal overfitting (controlled with Dropout)
- Model converged after ~80 epochs

---

## 📊 Model Evaluation

### Predictions vs Actual

![Predictions vs Actual](screenshots/03_predictions_vs_actual.png)

### Residual Analysis

![Residuals](screenshots/04_residual_plot.png)

**Key Insights:**
- Strong correlation between predictions and actual values
- Residuals randomly distributed around zero
- Model performs well across all price ranges

---

## 🔍 Data Preprocessing

**1. Missing Values:**
- Numerical: Median imputation
- Categorical: Most frequent imputation

**2. Encoding:**
- Label Encoding for categorical features (43 features)

**3. Scaling:**
- StandardScaler (mean=0, std=1) for all features

**4. Train-Validation Split:**
- 80% Training (1168 samples)
- 20% Validation (292 samples)

---

## 💡 Key Learnings

### PyTorch Fundamentals:
- ✅ Building custom nn.Module models
- ✅ Training loops with forward/backward passes
- ✅ DataLoader and batch processing
- ✅ Device management (CPU/GPU)

### Deep Learning Concepts:
- ✅ Backpropagation and gradient descent
- ✅ Activation functions (ReLU)
- ✅ Regularization (Dropout)
- ✅ Loss functions (MSELoss)
- ✅ Optimizers (Adam)

### Model Evaluation:
- ✅ RMSE calculation
- ✅ Overfitting detection
- ✅ Loss curve analysis
- ✅ Residual analysis

---

## 🎯 Comparison with ML Models

| Model | RMSE | Notes |
|-------|------|-------|
| Gradient Boosting (Week 2) | 0.1224 | Traditional ML |
| **PyTorch MLP (Week 3)** | **[SKORUN]** | Deep Learning |

**Insights:**
- [Karşılaştırma yapmak istersen ekle]

---

## 🔮 Future Improvements

**1. Architecture:**
- Try deeper networks (5-6 layers)
- Experiment with Batch Normalization
- Test different activation functions (LeakyReLU, ELU)

**2. Hyperparameters:**
- Learning rate scheduling
- Different optimizers (SGD with momentum, RMSprop)
- Varying dropout rates (0.1 - 0.5)

**3. Training:**
- Early stopping
- More epochs (200-300)
- Cross-validation

**4. Advanced Techniques:**
- Ensemble multiple models
- Feature engineering
- Data augmentation

---

## 📄 Assignment Requirements

This project fulfills the Week 3 Deep Learning assignment:

- ✅ PyTorch MLP implementation
- ✅ At least 2 hidden layers (4 layers total)
- ✅ ReLU activation functions
- ✅ MSELoss and Adam optimizer
- ✅ Train-Validation split (80-20)
- ✅ RMSE evaluation
- ✅ Kaggle submission
- ✅ Complete documentation

---

## 📝 Presentation

**Key Points:**
1. Problem: House price prediction using Deep Learning
2. Architecture: 4-layer MLP with dropout
3. Training: 100 epochs, Adam optimizer
4. Results: Kaggle RMSE [SKORUN]
5. Learnings: PyTorch, backpropagation, overfitting control

---

## 👤 Author

**Mustafa Haybat**

- LinkedIn: [linkedin.com/in/mustafa-haybat](https://linkedin.com/in/mustafa-haybat)
- GitHub: [@mustafa-haybat](https://github.com/mustafa-haybat)

---

## 🙏 Acknowledgments

- Kaggle for the House Prices dataset
- PyTorch team for the excellent framework
- AI Engineering course instructors

---

## ⭐ Star this repo if you found it helpful!
```
