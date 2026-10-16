# FuelWise: Weekly Petrol Planner + ML Lab

A Streamlit budget planner with a separately labeled synthetic machine-learning demonstration.

## Run on Windows

Open Command Prompt in this folder and run:

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Or use `python` instead of `py` if needed.

## Data limitations

The supplied CSV has 2,000 records, but all distances are 1 km. Fuel values follow distance/mileage almost exactly. The app generates synthetic multi-distance scenarios to demonstrate regression and a holdout evaluation. The model's performance on synthetic scenarios does not validate accuracy on real driving data. For weekly budgeting, the exact mathematical formula is used instead of the ML prediction.

## Next steps toward real-world ML

Gather actual trip distance, litres refuelled, vehicle type, traffic, route type, temperature, and driving conditions (with appropriate consent). Evaluate by holding out trips or vehicles, and compare against the formula baseline. Never present synthetic model metrics as real-world accuracy.

## Currency
All prices, budgets and cost charts are in Indian rupees (INR, ₹). The default ₹105/L is an editable example, not a live fuel price.
