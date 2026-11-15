import csv
import json


def write_json(path, value, *, atomic=True, allow_nan=True):
    destination = path.with_suffix(path.suffix + ".tmp") if atomic else path
    destination.write_text(json.dumps(value, indent=2, allow_nan=allow_nan) + "\n")

    if atomic:
        destination.replace(path)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
