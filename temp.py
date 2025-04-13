import uuid
import json

template = {
    'requestid': 'x',
    'path1': {'traceid': 'x', 'length': 53, 'turnleft': 4, 'turnright': 4, "safescore": 4.3},
    'path2': {'traceid': 'x', 'length': 52, 'turnleft': 4, 'turnright': 4, "safescore": 4.5},
    'best': 'path2',
    'reason': "shorter and safe than path1"

}

for i in range(1):
    template['requestid'] = str(uuid.uuid4()).replace('-', '')
    template['path1']['traceid'] = str(uuid.uuid4()).replace('-', '')
    template['path2']['traceid'] = str(uuid.uuid4()).replace('-', '')
    # print(json.dumps(template, ensure_ascii=False, indent=4))
    print(json.dumps(template))


data = {
    "requestid": "306a7b8e17b011f0b173e02be94eee68",
    "path1": {
        "traceid": "306a7b8f17b011f0b5d9e02be943ee63",
        "length": 589,
        "turnleft": 4,
        "turnright": 4,
        "safescore": 4.3
    },
    "path2": {
        "traceid": "306a7b9017b011f0af97e02be943ee62",
        "length": 549,
        "turnleft": 4,
        "turnright": 4,
        "safescore": 4.9
    }
}

# 使用 json.dumps 将数据转换为一行的 JSON 字符串
one_line_json = json.dumps(data)
print(one_line_json)