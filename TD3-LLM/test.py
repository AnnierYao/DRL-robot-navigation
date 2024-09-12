import json
import re

content = '''json
{
  "state_sequence": [
    [0.890, 0.808, 0.741, 0.708, 0.698, 0.694, 0.703, 0.719, 0.772, 0.851, 0.964, 1.152, 1.478, 2.112, 3.207, 3.214, 2.245, 2.287, 3.573, 3.916, 1.843, 1.442, 1.0, -0.850],
    [1.024, 0.839, 0.723, 0.661, 0.608, 0.581, 0.574, 0.572, 0.578, 0.607, 0.649, 0.707, 0.816, 0.986, 1.271, 1.847, 3.167, 3.169, 2.249, 2.258, 1.852, 1.836, 1.0, -0.549],
    [1.264, 0.887, 0.684, 0.572, 0.505, 0.464, 0.434, 0.419, 0.414, 0.414, 0.427, 0.451, 0.486, 0.541, 0.632, 0.781, 1.036, 1.633, 3.144, 3.156, 1.949, 2.256, 1.0, 0.318],
    [2.041, 0.924, 0.610, 0.455, 0.375, 0.320, 0.300, 0.301, 0.406, 0.394, 0.397, 0.404, 0.301, 0.300, 0.309, 0.361, 0.429, 0.554, 0.806, 1.548, 2.035, 2.476, 0.352, -0.282]
  ],
  "next_state_sequence": [
    [1.500, 0.900, 0.600, 0.500, 0.400, 0.350, 0.340, 0.330, 0.320, 0.310, 0.300, 0.290, 0.280, 0.270, 0.260, 0.250, 0.240, 0.230, 0.220, 0.210, 0.200, 0.190, 1.0, -0.300],
    [1.600, 0.850, 0.580, 0.520, 0.450, 0.400, 0.390, 0.380, 0.370, 0.360, 0.350, 0.340, 0.330, 0.320, 0.310, 0.300, 0.290, 0.280, 0.270, 0.260, 0.250, 0.240, 1.0, -0.250],
    [1.700, 0.800, 0.560, 0.540, 0.480, 0.420, 0.410, 0.400, 0.390, 0.380, 0.370, 0.360, 0.350, 0.340, 0.330, 0.320, 0.310, 0.300, 0.290, 0.280, 0.270, 0.260, 1.0, -0.200]
  ],
  "odom_x_sequence": [2.0395, 2.1148, 2.1724, 2.1773, 2.2000, 2.2200, 2.2400],
  "odom_y_sequence": [-4.551, -4.6618, -4.8649, -4.9881, -5.0000, -5.0200, -5.0400],
  "angle_sequence": [-0.8122, -1.1338, -1.4552, -1.6316, -1.7000, -1.7500, -1.8000],
  "goal_x": 2.1773,
  "goal_y": -4.9881
}
'''

# 使用正则表达式提取 JSON
match = re.search(r'\{.*\}', content.strip(), re.DOTALL)
if match:
    json_str = match.group()  # 提取到的 JSON 字符串
    try:
        # 解析 JSON 字符串
        response_dict = json.loads(json_str)
        print(response_dict)  # 处理解析后的字典
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
else:
    print("No JSON object found")


# # 提取 goal_x 和 goal_y
state_sequence = response_dict.get('state_sequence')
next_state_sequence = response_dict.get('next_state_sequence')
odom_x_sequence = response_dict.get('odom_x_sequence')
odom_y_sequence = response_dict.get('odom_y_sequence')
angle_sequence = response_dict.get('angle_sequence')
goal_x = response_dict.get('goal_x')
goal_y = response_dict.get('goal_y')

print('new goal', goal_x, goal_y)
print('state sequence', state_sequence)
print('next state sequence', next_state_sequence)
print('odom x sequence', odom_x_sequence)
print('odom y sequence', odom_y_sequence)
print('angle sequence', angle_sequence)



{
  "walls": [
    {"start": [-9, -9], "end": [-9, 9]},
    {"start": [-9, -9], "end": [9, -9]},
    {"start": [-9, 9], "end": [9, 9]},
    {"start": [9, -9], "end": [9, 9]},
    {"start": [0, 7.5], "end": [0, 2.8]},
    {"start": [0, 2.8], "end": [9, 2.8]},
    {"start": [-3, 4], "end": [-3,  -1]},
    {"start": [-3, -1], "end": [9, -1]},
    {"start": [-6, 6], "end": [-6, -6]},
    {"start": [-6, -6], "end": [9, -6]},
    {"start": [-3, 9], "end": [-3, 7.5]}
  ]
}
