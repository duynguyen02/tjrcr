# TJRCR 1.0.0

Library for validating parameters for comprehensive reservoir regulation

### Installation

```
pip install tjrcr
```

### Getting Started

#### Required Data

For detailed data requirements, see the [TJWB](https://github.com/duynguyen02/tjwb) library. Reservoir data should span
at least 10 years and include complete data for all 12 months each year for effective calculations.

#### Usage

```python
import pandas as pd
from tjwb import WB

from tjrcr import TJRCR

pre_df = pd.read_csv('data_example.csv')
ets_df = pd.read_csv('elevation_to_storage_example.csv')

pre_df['boxdrain_cong1'] = pre_df['boxdrain_cong1'] / 100  # centimeters to meters
pre_df['valveoverflow_tran1'] = pre_df['valveoverflow_tran1'] / 100  # centimeters to meters
pre_df['valveoverflow_tran2'] = pre_df['valveoverflow_tran2'] / 100  # centimeters to meters
pre_df['valveoverflow_tran3'] = pre_df['valveoverflow_tran3'] / 100  # centimeters to meters
pre_df['timestamp'] = pd.to_datetime(pre_df['timestamp'], format='%d/%m/%Y %H:%M')  # convert to valid format

# Details in the TJWB library
tjwb = WB(
    ets_df,
    boxdrain_elevation=22.5,
    boxdrain_height=0.7,
    valveoverflow_elevation=22,
    valveoverflow_height=3
)

tjrcr = TJRCR(
    tjwb,
    ets_df=ets_df,
    V_c=0.66,  # dead water level
    V_h=2.22  # usable water level
)

result = tjrcr.is_comprehensive_regulation(
    pre_df=pre_df,
    eps=0.022,
    P=0.95,
    round_to=3,  # round results in water balance calculations
    forced_gt_10_year=False,  # require data set to be at least 10 years
    forced_12_months_each_year=True,  # require data set to have complete 12 months each year
    forced_elevation=False  # remove invalid data columns during water balance calculations
)

if result:
    print("Parameters are valid for comprehensive regulation")
else:
    print("Parameters are not valid for comprehensive regulation")
```

## Changelog

### Version 1.0.0 - Initial Release - July 21, 2024

- Initial release with core functionalities

Each section in this changelog provides a summary of what was added, changed, fixed, or removed in each release, helping
users and developers understand the evolution of the project and highlighting important updates or improvements.