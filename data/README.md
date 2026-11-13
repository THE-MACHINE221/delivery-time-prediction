# Data

This project uses the delivery dataset from the **KAUST Academy AI program** (Kaggle dataset
`mohammad2012191/q1-ka-ai-2026`). It is a modified copy of a public Kaggle dataset of 1,000 food
orders: the KAUST version adds duplicate rows and extra missing values.

The CSV is not stored in this repository. To download it into this folder:

```python
import kagglehub, shutil
path = kagglehub.dataset_download("mohammad2012191/q1-ka-ai-2026")
shutil.copy(f"{path}/Q1_data.csv", "data/Q1_data.csv")
```

Columns: `Order_ID`, `Distance_km`, `Weather`, `Traffic_Level`, `Time_of_Day`, `Vehicle_Type`,
`Preparation_Time_min`, `Courier_Experience_yrs`, and the target `Delivery_Time` (minutes).
