import uuid
import json

with open('dataset/case.json', 'r', encoding='utf-8') as fr:
    cases = json.load(fr)

cases_txt = []
for template in cases:
    template['requestid'] = str(uuid.uuid4()).replace('-', '')
    template['path1']['traceid'] = str(uuid.uuid4()).replace('-', '')
    template['path2']['traceid'] = str(uuid.uuid4()).replace('-', '')
    print(json.dumps(template))
    cases_txt.append(template)

with open('dataset/cases.txt', 'r') as fw:
    fw.writelines(cases_txt)

data = {
    "requestid": "306a7b8e17b011f0b173e02be94eee68",
    "path1": {
        "traceid": "306a7b8f17b011f0b5d9e02be943ee63",
        "length": 1783.59,
        "safescore": 4.3
    },
    "path2": {
        "traceid": "306a7b9017b011f0af97e02be943ee62",
        "length": 1549.53,
        "safescore": 4.9
    }
}

# 使用 json.dumps 将数据转换为一行的 JSON 字符串
one_line_json = json.dumps(data)
print(one_line_json)