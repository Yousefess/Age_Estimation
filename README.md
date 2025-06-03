# 📦 Age Estimation from Face Images using PyTorch

This project implements an **age estimation** model using deep learning in **PyTorch**. It uses the **UTKFace dataset** and a **ResNet-50** backbone for predicting a person's age from a facial image as a regression task.

---

## 📁 Project Structure

```arduino
.
├── config.py
├── main.py
├── inference.py
├── data/
│   └── create_csv_files.py
│   └── custom_dataset.py
│   └── show_random_samples.py
│   └── split_data.py
├── models/
│   └── age.py
├── notebook/
│   └── age_estimation.ipynb
├── train/
│   ├── train_and_evaluation.py
├── utils/
│   └── helpers.py
└── README.md
```

---

## 🚀 Features

- 🔥 Age estimation using ResNet-50
- 📦 Modular codebase (easy to extend and debug)
- 🧪 Train/Validation pipeline
- 📊 MSE loss for regression
- ✅ Easily switch model or dataset
- ⚙️ Clean config system

---

## 📚 Dataset

The project uses the [UTKFace dataset](https://susanqq.github.io/UTKFace/) — a large-scale face dataset with age, gender, and ethnicity labels. Each filename follows the pattern:

```css
[age]_[gender]_[ethnicity]_[date&time].jpg
```

Only the `age` label is used in this project.

> 📁 Place the dataset inside a `data/UTKFace/` directory, so the path looks like:
>
> ```kotlin
> data/UTKFace/25_1_2_20170116174525125.jpg.chip.jpg
> ```

---

## 🧰 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

All configuration values are in `config.py`.

---

### 🏁 How to Run

Train the model:

```bash
python main.py
```

---

### 🧪 Evaluation

Validation MSE is printed after each epoch. Lower is better.

> Want to evaluate separately? Run `evaluate(model, val_loader, loss_fn, device)` from `train/evaluate.py`.
