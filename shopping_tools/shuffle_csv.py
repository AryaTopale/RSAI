import random


def shuffle_csv(input_file, output_file):
    header = None
    rows = []
    with open(input_file, "r") as f:
        header = f.readline().strip()
        for line in f:
            rows.append(line.strip())

    random.Random(42).shuffle(rows)

    with open(output_file, "x") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")


if __name__ == "__main__":
    input_csv = "shopping_tools/shopping_tool_positives_raw.csv"
    output_csv = "shopping_tools/shopping_tool_positives.csv"
    shuffle_csv(input_csv, output_csv)
    print(f"Shuffled CSV saved to {output_csv}")
