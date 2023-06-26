# Tools

## `count.py`

Summarize and count csv files generated from the MOT.

```bash
python3 count.py --output-csv all_counts.csv --upstream right /path/to/countables/folder
```
Short form:
```bash
python3 count.py -o all_counts.csv -u right /path/to/countables/folder
```

`--upstream` denotes the direction that the fish must swim towards to count
as one count.

## `count_combine.py` Combining MOT and Class Detections

Make sure all your MOT tracks CSV files have the suffix `tracks.csv`.
```bash
python3 count.py --mot-folder mot_folder --det-folder det_folder
```
