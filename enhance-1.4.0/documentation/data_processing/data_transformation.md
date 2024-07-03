## Data Transformation

### Normalization
The `normalize` method in `DataTransformer` class is used to normalize specified columns using various scaling methods.

**Arguments:**
- `df (DataFrame)`: DataFrame containing the data.
- `input_cols (list)`: List of columns to be normalized.
- `output_col (str)`: Name of the output column for normalized features.
- `method (str)`: Normalization method, one of 'min-max', 'standard', 'max-abs', 'robust'.

**Returns:**
- `DataFrame`: DataFrame with normalized columns.

**Example:**
Supported methods: 'min-max', 'standard', 'max-abs', 'robust'.
```python
from data_processing.data_transformation.transformation import DataTransformer

df = DataTransformer.normalize(df, input_cols=['col1', 'col2'], output_col='scaled_features', method='robust')
```