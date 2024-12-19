if __name__ == "__main__":
    import json

    json_file_path = "rules.json"
    # 存储键值对的 JSON 文件

    with open(json_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    sorted_data = {k: data[k] for k in sorted(data.keys(), key=lambda k: len(k), reverse=True)}

    print("按照键长度降序排序后的结果：")
    print(json.dumps(sorted_data, ensure_ascii=False, indent=4))

    output_file_path = "rules_sorted.json"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(sorted_data, output_file, ensure_ascii=False, indent=4)
        print(f"排序后的 JSON 文件已保存到: {output_file_path}")

