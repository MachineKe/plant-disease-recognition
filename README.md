# Plant Disease Recognition Project

This project implements a deep learning pipeline for classifying plant diseases into three categories: **Healthy**, **Powdery**, and **Rust** using transfer learning (MobileNetV2).

## Environment Setup

## Install Python 3.11: Ensure you have Python 3.11.x installed on your system.

### Step 1: Create Virtual Environment

Open your terminal or command prompt in the project directory and run:

```bash
python -m venv plant-disease-env
```

### Step 2: Activate Virtual Environment

- **On Command Prompt:**

```bash
plant-disease-env\Scripts\activate.bat
```

- **On PowerShell:**

```powershell
.\plant-disease-env\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Jupyter Notebook

After installing dependencies, launch the Jupyter notebook:

```bash
jupyter notebook train_model.ipynb
```

This will open the notebook in your default web browser, allowing you to execute each step interactively.

## Running Flask GUI

After installing dependencies, launch the Flask GUI:

```cmd
cd web_app
run_app.bat
```

This will start the Flask application, and you can access it at `http://localhost:8000`.

## Running GUI with WSGI

```cmd
cd web_app
waitress-serve --listen=127.0.0.1:8000 app:app
```

This will run GUI in production mode on port 8000.

## Project Structure

- `data/`: Contains training, validation, and test datasets.
- `train_model.ipynb`: Jupyter notebook for interactive model training and evaluation.
- `train_model.py`: Python script version of the notebook.
- `requirements.txt`: Python dependencies required for the project.
# plant-disease-recognition
# plant-disease-detection-mono
