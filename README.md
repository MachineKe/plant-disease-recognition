# Plant Disease Recognition Project

This project implements a deep learning pipeline for classifying plant diseases into three categories: **Healthy**, **Powdery**, and **Rust** using transfer learning (MobileNetV2).

## Environment Setup

### Using Python 3.11 on Linux (Ubuntu)

1. **Check your current Python version:**
   ```bash
   python3 --version
   ```
   If the output is not `Python 3.11.x`, proceed to install Python 3.11.

2. **Install Python 3.11 and required packages:**
   ```bash
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3.11-dev
   ```

3. **Set up the virtual environment using Python 3.11:**
   ```bash
   python3.11 -m venv plant-disease-env
   ```

> **Important:** Always use `python3.11` and `pip` from the `plant-disease-env` virtual environment to ensure you are using the correct Python version for all steps below.

### Step 1: Create Virtual Environment

Open your terminal or command prompt in the project directory and run:

```bash
python -m venv plant-disease-env
```

### Step 2: Activate Virtual Environment

- **On Windows (Command Prompt):**

```bash
plant-disease-env\Scripts\activate.bat
```

- **On Windows (PowerShell):**

```powershell
.\plant-disease-env\Scripts\Activate.ps1
```

- **On Linux (e.g., Ubuntu):**

```bash
source plant-disease-env/bin/activate
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

- **On Windows:**

```cmd
cd web_app
run_app.bat
```

- **On Linux:**

```bash
cd web_app
python app.py
```

This will start the Flask application, and you can access it at `http://localhost:8000`.

## Running GUI with WSGI

- **On Windows:**

```cmd
cd web_app
waitress-serve --listen=127.0.0.1:8000 app:app
```

- **On Linux:**

First, install gunicorn:

```bash
pip install gunicorn
```

Then run:

```bash
cd web_app
gunicorn --bind 0.0.0.0:8000 app:app
```

This will run the GUI in production mode on port 8000.

## Project Structure

- `data/`: Contains training, validation, and test datasets.
- `train_model.ipynb`: Jupyter notebook for interactive model training and evaluation.
- `train_model.py`: Python script version of the notebook.
- `requirements.txt`: Python dependencies required for the project.
# plant-disease-recognition
# plant-disease-detection-mono
