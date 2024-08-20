import glob
import base64
from mimetypes import guess_type
# from PIL import Image
from openai import AzureOpenAI
# import ast
import re
import json


class GoalPredictor:
    def __init__(self, api_base, api_key, deployment_name, api_version):
        self.api_base = api_base
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.client = self._create_client()

    def _create_client(self):
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            base_url=f"{self.api_base}/openai/deployments/{self.deployment_name}"
        )


    # def local_image_to_data_url(self, image_path):
    #     mime_type, _ = guess_type(image_path)
    #     if mime_type is None:
    #         mime_type = 'application/octet-stream'

    #     with open(image_path, "rb") as image_file:
    #         base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    #     return f"data:{mime_type};base64,{base64_encoded_data}"

    # def get_new_goal(self, goal_x, goal_y, state_sequence, odom_sequence):

    #     # state_sequence = []
    #     # with open('state_squence', 'r') as f:
    #     #     state_sequence = f.read()
    #     goal = [goal_x, goal_y]
    #     response_dict = "{'goal_x': x, 'goal_y': y}"
    #     formatted_string = '''Give me a possible goal point to make the existing trajectory have the highest reward without doing other changes in this history data. please only return me a goal coordinate and The response should strictly look like: '{}'. 
    #     The goal in this trajectory is (goal_x, goal_y): {}. Given the last 50 sequence of states (where the car didn't achieve the goal): {}. Plus the last 50 trajectories of the car, [odom_x, odom_y]: {}.'''.format(response_dict, goal, state_sequence, odom_sequence)

    #     response = self.client.chat.completions.create(
    #         model=self.deployment_name,
    #         messages=[
    #             { "role": "system", "content": "You are a helpful assistant." },
    #             { "role": "user", "content": [  
    #                 { 
    #                     "type": "text", 
    #                     "text": '''I have a robotic car. A target position is given in 3D Cartesian space.
    #                                 This task has the following goals:
    #                                 1. The car should drive to the target position and stay at that position (The
    #                                 normalized distance on the x and y axes between the robot and the target position should be
    #                                 less than 0.4).
    #                                 2. The car cannot have a collision (the minimum lidar point distance cannot be
    #                                 lower than 0.35).
    #                                 The reward function and state are defined as follows:
    #                                 1. A large positive reward (100.0) for reaching the goal.
    #                                 2. A large negative penalty (-100.0) for collisions.
    #                                 3. A calculated reward based on speed, turning rate, and proximity to obstacles for regular navigation, encouraging faster, straighter paths and safer distances from obstacles：
    #                                     r = linear_velocity.x / 2 - abs(angular_velocity.z) / 2 - r3(min_laser) / 2, where r3 = lambda x: 1 - x if x < 1 else 0.0 (min_laser is the minimum distance reading from the Velodyne laser sensor).
    #                                 4. The state is a array shaped[24,], including laser data[1,20] and robot_state = [distance to goal, theta between goal, linear_velocity.x, angular_velocity.z].
    #                                 5. The laser data is a 1D array of 20 float values, representing the distance to the nearest obstacle in 20 directions, the direction gaps are from -π/2 to π/2.
    #                                 6. The gaps are [[-1.60, -1.41], [-1.41, -1.25], [-1.25, -1.10], [-1.10, -0.94], [-0.94, -0.78], [-0.78, -0.62], [-0.62, -0.47], [-0.47, -0.31], [-0.31, -0.15], [-0.15, 0.0], [0.0, 0.15], [0.15, 0.31], [0.31, 0.47], [0.47, 0.62], [0.62, 0.78], [0.78, 0.94], [0.94, 1.10], [1.10, 1.25], [1.25, 1.41], [1.41, 1.60]]'''
    #                 }
    #             ] } ,

    #             { "role": "user", "content": [  
    #                 { 
    #                     "type": "text",
    #                     "text": formatted_string
    #                 }
    #         ] },
    #         ],
    #         max_tokens=100,
    #         temperature=0,
    #     )

    def get_new_goal(self, goal_x, goal_y, reason, odom_state_sequence):

        # state_sequence = []
        # with open('state_squence', 'r') as f:
        #     state_sequence = f.read()
        goal = [goal_x, goal_y]
        response_dict = "{'goal_x': x, 'goal_y': y}"
        prompt_string = '''
            Given the context of reinforcement learning and hindsight experience replay, provide a new goal point that maximizes the reward for the given trajectory. The goal should be achievable given the robot’s final position and the provided trajectory.

            Instructions:
            1. The goal should be feasible based on the robot’s final position and should lie within the environment's bounds.
            2. The objective is to reuse this trajectory and generate the highest possible reward.
            3. The goal should not very close to the collision point, otherwise it will lead the robot to make collisions.

            Details:
            - The original goal in this trajectory is (goal_x, goal_y): {}.
            - The robot did not achieve this goal, due to {}.
            - The last 50 sequence of steps ([robot_odom_x, robot_odom_y, distance to goal, min_laser_distance, theta between goal, linear_velocity.x, angular_velocity.z]) in this trajectory are: {}.

            Please only return me a goal coordinate and The response should strictly look like: '{}'.
            '''.format(goal, reason, odom_state_sequence, response_dict)


        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": '''I have a robotic car. A target position is given in 3D Cartesian space.
                                    This task has the following goals:
                                    1. The car should drive to the target position (The
                                    normalized distance on the x and y axes between the robot and the target position should be
                                    less than 0.3).
                                    2. The car cannot have a collision (the minimum lidar point distance (min_laser_distance) cannot be
                                    lower than 0.35).
                                    The reward function is:
                                    1. A large positive reward (100.0) for reaching the goal.
                                    2. A large negative penalty (-100.0) for collisions.
                                    3. A calculated reward based on speed, turning rate, and proximity to obstacles for regular navigation, encouraging faster, straighter paths and safer distances from obstacles：
                                        r = linear_velocity.x / 2 - abs(angular_velocity.z) / 2 - r3(min_laser) / 2, where r3 = lambda x: 1 - x if x < 1 else 0.0 (min_laser is the minimum distance reading from the Velodyne laser sensor).
                                    '''
                    }
                ] } ,

                { "role": "user", "content": [  
                    { 
                        "type": "text",
                        "text": prompt_string
                    }
            ] },
            ],
            max_tokens=300,
            temperature=0,
        )

        content = response.choices[0].message.content
        # print(content)
        # print('new goal',response)
        # score_match = re.search("{'score': (\d+)}", content)
        # 解析为字典
        # response_dict = ast.literal_eval(content)
        json_str = re.search(r'\{.*?\}', content).group()

        # 将单引号替换为双引号
        json_str = json_str.replace("'", '"')

        # 使用 json.loads 解析为字典
        response_dict = json.loads(json_str)

        # 提取 goal_x 和 goal_y
        goal_x = response_dict.get('goal_x')
        goal_y = response_dict.get('goal_y')
   

        return goal_x, goal_y
    


# api_base = "https://ai-azureaiwestusengine881797519629.openai.azure.com/"
# api_key = '4d44030e3b2b4f978a44cccc49b464f1'
# deployment_name = 'gpt-4'
# api_version = '2023-03-15-preview'

# image_scorer = ImageScorer(api_base, api_key, deployment_name, api_version)
# map_index = 1950
# score = image_scorer.get_score_for_images(map_index)
