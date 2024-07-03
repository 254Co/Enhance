## Feature Engineering

### Polynomial Features
The `polynomial_features` method in `FeatureEngineer` class is used to generate polynomial features for specified columns.

**Arguments:**
- `df (DataFrame)`: DataFrame containing the data.
- `columns (list)`: List of columns to generate polynomial features for.
- `degree (int)`: Degree of the polynomial features.

**Returns:**
- `DataFrame`: DataFrame with polynomial features added.

**Example:**
```python
from data_processing.data_transformation.feature_engineering import FeatureEngineer

df = FeatureEngineer.polynomial_features(df, columns=['feature1', 'feature2'], degree=2)
```