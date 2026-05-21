# Dataset Layout

Datasets are split into local training data and publishable test data.

```text
data/train_data/                         # 50-case training CSV dataset
data/test_data/                          # Publishable test_3MW/test_6MW/test_8MW CSV cases
data/psm/train_data/                     # a_1MW through j_1MW PyroSim source models
data/psm/test_data/                      # fst_3MW/fst_6MW/fst_8MW PyroSim source models
data/raw/                               # Local raw archives and generated simulation outputs; ignored
data/sample/                            # Legacy local leftovers; ignored
```

`data/train_data` contains 50 cases and uses the format expected by `src.data.dataset.myDataset`: each case folder contains one `{case}_devc.csv` file and 150 `slice_*.csv` files.

`data/test_data` uses the same loader format with three cases named `test_3MW`, `test_6MW`, and `test_8MW`.

All `.psm` files are centralized in `data/psm` so they do not get mixed into the CSV datasets.
