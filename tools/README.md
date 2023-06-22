# Tools

## count.py

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
