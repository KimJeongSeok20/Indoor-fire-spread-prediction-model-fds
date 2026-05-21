# Fire Surface Test Data

PyroSim/FDS test cases generated from the existing 7MW PyroSim model, plus model-ready CSV exports for evaluation.
Only the `Fire` surface `HRRPUA` and the `FIRE` obstruction `XB` were changed for the `.psm` models.
The selected fire locations avoid the two extracted original left-edge fire ranges currently visible in this workspace.

Layout:

```text
inputs/
  fst_3MW/
  fst_6MW/
  fst_8MW/
model_ready/
  test_3MW/
  test_6MW/
  test_8MW/
../../psm/test_data/
  fst_3MW.psm
  fst_6MW.psm
  fst_8MW.psm
```

Each `inputs/fst_*` folder includes:

- `{case}.fds`: matching 150 second FDS input exported/generated for reference.
- `psm_fire.jpg` and `psm_blue.jpg`: texture files referenced by the model.

The matching PyroSim `.psm` files live in `data/psm/test_data`.

Each `model_ready/test_*` folder includes one `{case}_devc.csv` file and 150 `slice_*.csv` files for `src.data.dataset.myDataset`.

| Input case | Model case | MW | HRRPUA | Width | Depth | Area | FIRE XB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fst_3MW | test_3MW | 3 | 3000.0 | 3.0 m | 4.0 m | 12.0 m2 | `22.75,25.75,12.25,16.25,0.1,0.5` |
| fst_6MW | test_6MW | 6 | 6000.0 | 5.0 m | 3.5 m | 17.5 m2 | `40.5,45.5,30.5,34.0,0.1,0.5` |
| fst_8MW | test_8MW | 8 | 8000.0 | 6.0 m | 4.0 m | 24.0 m2 | `51.25,57.25,39.25,43.25,0.1,0.5` |

The `slice_*.csv` files were exported with `fds2ascii` from SLCF output using sampling factor `2`, domain `y`, bounds `0 65 0 52 2.5 2.5`, 1-second windows from `0-1` through `149-150`, and variables `TEMPERATURE`, `SOOT_VISIBILITY`, and `CO_FRACTION`.
