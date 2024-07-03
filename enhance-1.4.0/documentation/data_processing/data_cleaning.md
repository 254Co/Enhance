## Data Cleaning

### Impute Missing Values
The `impute_missing_values` method in `AdvancedDataCleaner` class is used to impute missing values in specified columns using a given strategy.

**Arguments:**
- `df (DataFrame)`: DataFrame containing the data.
- `columns (dict)`: Dictionary with column names as keys and impute values as values.
- `strategy (str)`: Strategy for imputing missing values, either 'mean' or 'median'.

**Returns:**
- `DataFrame`: DataFrame with imputed values.

**Example:**
```python
from data_processing.data_transformation.advanced_data_cleaning import AdvancedDataCleaner

df = AdvancedDataCleaner.impute_missing_values(df, columns=['col1', 'col2'], strategy='mean')
```

### Remove Outliers
The `remove_outliers` method in `AdvancedDataCleaner` class is used to remove outliers in specified columns based on quantile bounds.

**Arguments:**
- `df (DataFrame)`: DataFrame containing the data.
- `columns (list)`: List of columns from which to remove outliers.
- `lower_bound (float)`: Lower quantile bound.
- `upper_bound (float)`: Upper quantile bound.

**Returns:**
- `DataFrame`: DataFrame with outliers removed.

**Example:**
```python
from data_processing.data_transformation.advanced_data_cleaning import AdvancedDataCleaner

df = AdvancedDataCleaner.remove_outliers(df, columns=['col1', 'col2'], lower_bound=0.05, upper_bound=0.95)
```