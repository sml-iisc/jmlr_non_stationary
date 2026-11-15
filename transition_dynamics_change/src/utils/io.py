import csv
import json


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path, rows):
    if not rows:
        return

    fields = list(dict.fromkeys((k for row in rows for k in row)))
    temporary = path.with_suffix(".tmp")

    with temporary.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)
