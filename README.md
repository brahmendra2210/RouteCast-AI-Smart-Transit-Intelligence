# BusBoard AI — runnable project

A runnable implementation of the uploaded project topic: **predicting bus boarding classes from smart-card records while addressing class imbalance**.

## What is included

- CSV upload and a generated demo dataset
- Date feature extraction (`year_value`, `month`, `day`, `hour`, `minute`, `second`)
- Missing-value handling, categorical encoding and feature scaling
- A small conditional tabular GAN to synthesize minority-class training records
- Baselines: DNN on original data and SMOTE + DNN
- Proposed model: GAN + DNN
- Extension: GAN + CNN2D
- Accuracy, macro precision, macro recall, macro F1, confusion matrices and downloadable predictions

## Requirements

- Windows 10/11, macOS or Linux
- Python 3.10 or 3.11 (recommended)
- Internet for the first dependency installation
- 8 GB RAM minimum; 16 GB recommended
- CPU execution is supported; a GPU is not required

## Run on Windows

1. Extract the ZIP to a normal folder (not inside the ZIP preview).
2. Install Python 3.10 or 3.11 from https://www.python.org/downloads/ and select **Add Python to PATH**.
3. Double-click `run_windows.bat`.
4. Wait for package installation. Streamlit opens the application in your browser, usually at `http://localhost:8501`.
5. Select **Use generated demo dataset** and click **Run complete experiment**.

First installation can take several minutes because PyTorch is a large dependency. Later launches are faster.

## Run manually (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in that terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Use your real dataset

Upload a CSV with at least 50 rows. Required columns:
- `NumberOfBoardings` — target class/label
- `WeekBeginning` — date/time used to derive calendar features, or provide an `Hour` column instead

`StopName` is supported and encoded as a categorical feature. Other numeric and categorical columns are also accepted. The supplied document's code reads `Dataset/BusBoard.csv`, encodes `StopName`, extracts date components from `WeekBeginning`, and uses `NumberOfBoardings` as its target.

**Important:** The uploaded report did not include the actual `BusBoard.csv` smart-card data or a trained DC-GAN checkpoint. The project therefore includes a clearly labelled synthetic demo dataset so the application runs immediately. Upload authorized real data for real-world results.

## Methodology and interpretation

The app performs a stratified train/test split first. Preprocessing is fitted only on the training partition. GAN and SMOTE resampling are applied only to training data; the untouched test set is used for evaluation. This avoids the test-set contamination present in many naive resampling workflows.

The GAN is a compact conditional fully connected GAN for tabular records, not an image DCGAN. The CNN2D extension reshapes tabular features into a one-row feature map. This is an engineering adaptation of the report's stated GAN/DNN/CNN2D workflow, not a claim of exact reproduction of the original authors' architecture.

The report's screenshots claim approximately 39% for SMOTE+DNN, 90% for proposed GAN+DNN and 91% for CNN2D extension. Those are report-specific results and are **not guaranteed or hard-coded** here. This implementation calculates metrics from the dataset you run.

## Troubleshooting

- `python is not recognized`: reinstall Python and select Add Python to PATH.
- `pip` errors: run `python -m pip install --upgrade pip`, then retry.
- Port 8501 busy: run `streamlit run app.py --server.port 8502`.
- PyTorch installation is slow: allow time for the initial download; use a stable internet connection.
- Memory pressure: reduce GAN epochs and synthetic samples in the sidebar.
- CSV column error: check exact target name `NumberOfBoardings` and date column `WeekBeginning` or numeric `Hour`.

## Project structure

```text
BusBoard_AI_Project/
├── app.py
├── requirements.txt
├── run_windows.bat
├── run_linux_mac.sh
├── README.md
└── src/
    ├── __init__.py
    └── pipeline.py
```
