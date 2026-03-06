import csv


def filter_metatool_data(description=False):
    input_file = "metatool_dataset_with_descriptions.csv"
    target_tools = {
        "ProductSearch",
        "Discount",
        "Review",
        "ProductComparison",
        "ShoppingAssistant",
    }
    output_file = "shopping_tools/shopping_tool_positives.csv"

    try:
        with open(output_file, mode="w", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            if description:
                writer.writerow(["Query", "Tool", "Tool_Description"])
            else:
                writer.writerow(["Query", "Tool"])
            with open(input_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # Print the header
                if reader.fieldnames:
                    print(",".join(reader.fieldnames))

                count = 0

                # Print matching rows
                for row in reader:
                    if row["Tool"] in target_tools:
                        count += 1
                        if description:
                            writer.writerow(
                                [row["Query"], row["Tool"], row["Tool_Description"]]
                            )
                        else:
                            writer.writerow([row["Query"], row["Tool"]])
        print(f"Total matching rows: {count}")

    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    filter_metatool_data()
