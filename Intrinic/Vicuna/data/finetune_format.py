import json

def read_file_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

def instruction(inputs):
    instruction = f"Given several concepts: \"{inputs}\", write a coherent sentence as short as possible using background commonsense knowledge: "
    return instruction

def create_conversation_data(human_line, gpt_line):
    conversations = [{"from": "human", "value": instruction(human_line)}]
    if isinstance(gpt_line, list):
        conversations.append({"from": "gpt", "value": gpt_line[0]})
    else:
        conversations.append({"from": "gpt", "value": gpt_line})
    return conversations

def main():
    human_lines = read_file_lines('train.src.txt')
    gpt_lines = read_file_lines('train.tgt.txt')

    data = []
    for i in range(len(human_lines)):
        data.append({
            "id": f"commongen_{i}",
            "conversations": create_conversation_data(human_lines[i:i+1], gpt_lines[i:i+1])
        })

    with open('train.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()