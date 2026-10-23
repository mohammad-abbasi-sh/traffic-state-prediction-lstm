# Data folder

Not included in the repo. Each dataset is expected to be a folder named
`DataSet-<n>` containing one or more headerless CSV files, e.g.:

```
data/
  DataSet-1/
    part1.csv
    part2.csv
  DataSet-2/
    ...
  DataSet-3/
    ...
```

Columns 0-7: traffic state features (e.g. queue lengths per approach).
Columns 8 onward (last 4 columns): signal action / phase configuration.

Update `DATASET` and `DIR_PATH` at the top of the scripts in `src/` to point
at the dataset you want to run.
