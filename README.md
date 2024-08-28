# tjrcr

`tjrcr` is a Python library designed for validating parameters for comprehensive reservoir regulation. This library
provides methods to ensure that the input data meets specific criteria required for effective reservoir management, such
as having a full 12 months of data for each year and a minimum of 10 years of data.

## Installation

To install the library, use pip:

```bash
pip install tjrcr
```

## Usage

### Basic Example

```python
import pandas as pd
from tjrcr import is_comprehensive_regulation
from tjwb import TJWBResult  # This is referenced from the tjwb library

# Prepare your data
df = pd.DataFrame({
    'datetime': pd.date_range(start='2010-01-01', periods=120, freq='M'),
    'inflow_speed': range(120),
    'outflow_speed': range(120, 240)
})

# This is just an example for the TJWBResult class, please refer to the tjwb library for more details
tjwb_result = TJWBResult(
    datetime=df['datetime'],
    inflow_speed=df['inflow_speed'],
    outflow_speed=df['outflow_speed'],
    components_outflow_speed={}
)

# Validate parameters
result = is_comprehensive_regulation(
    tjwb_result=tjwb_result,
    eps=1.0,
    P=80.0,
    V_c=1000.0,  # Dead water level
    forced_gt_10_year=False,  # Allows the function to proceed even if your dataset spans fewer than 10 years.
    forced_12_months_each_year=False
    # Allows the function to proceed even if any year in your dataset has fewer than 12 months of data.
)

print(f"Is comprehensive regulation valid? {result}")
```

### Functions

- **`is_comprehensive_regulation`**: Validates whether the input data is suitable for comprehensive reservoir regulation
  based on certain criteria like minimum years of data, full monthly data per year, and comparison against a threshold
  value.

## Error Handling

The `tjrcr` library includes validation steps that raise exceptions if the input data does not meet the required
criteria:

- **Insufficient Years of Data**:
    - If the dataset does not span at least 10 years, a `ValueError` is raised.

- **Incomplete Monthly Data**:
    - If any year in the dataset does not have data for all 12 months, a `ValueError` is raised.

## Reference to `tjwb`

The `tjrcr` library references the `TJWBResult` class from the `tjwb` library, which handles the conversion of reservoir
data into a format suitable for analysis. You can find more details on how to use `TJWBResult` and related calculations
in the [tjwb GitHub repository](https://github.com/duynguyen02/tjwb).

## License

This library is released under the MIT License.

## Contact

If you have any questions or issues, please open an issue on [GitHub](https://github.com/duynguyen02/tjrcr/issues) or
email us at [duynguyen02.dev@gmail.com](mailto:duynguyen02.dev@gmail.com).
