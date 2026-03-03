import json
# test_json2 =
#
# print(test_json2["messages"][0]["content"])
# print("--------------SEPARATOR-----------------")
# print(test_json2["messages"][1]["content"])


with open('LoRA_samples_intention_temp.jsonl', 'r', encoding='utf-8') as file:
    for line_num, line in enumerate(file, 1):
        try:
            data = json.loads(line.strip())

            print(data["messages"][0]["content"])
            print("--------------SEPARATOR-----------------")
            print(data["messages"][1]["content"])

            print()
            print("=============SECTION SEPARATOR=================")
            print()

        except json.JSONDecodeError as e:
            print(f"Error on line {line_num}: {e}")