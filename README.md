# FUTURE_ML_01 — Sales & Demand Forecasting

## Internship Task
Machine Learning — Future Interns, Task 1.

## Objective
Build a model to forecast future sales/demand from historical business data, with data cleaning, time-based feature engineering, model evaluation, and business-friendly visualizations.

## Project Structure
- `sales_data.csv` — reproducible synthetic daily sales dataset
- `sales_forecasting.ipynb` — complete analysis notebook
- `sales_forecasting.py` — standalone Python implementation
- `requirements.txt` — Python dependencies

## Method
1. Load and inspect historical daily sales.
2. Create calendar features.
3. Create lag and rolling-average features.
4. Split the data chronologically (80/20) to avoid random time leakage.
5. Train a Random Forest regression model.
6. Evaluate using MAE, RMSE and MAPE.
7. Visualize actual vs forecast sales.
8. Inspect feature importance and translate results into business actions.

## Business Use
Forecasts can support inventory planning, staffing, promotions and operational decisions.

## Dataset Note
The included dataset is synthetic and was generated for this internship demonstration. It is intentionally self-contained so anyone reviewing the repository can reproduce the analysis.

## Run Locally
```bash
pip install -r requirements.txt
jupyter notebook
```

Open `sales_forecasting.ipynb` and run all cells.

Or:
```bash
python sales_forecasting.py
```

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Jupyter Notebook.

## Author
Your Name
