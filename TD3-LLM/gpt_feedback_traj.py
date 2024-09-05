import glob
import base64
from mimetypes import guess_type
# from PIL import Image
from openai import AzureOpenAI
import numpy as np
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

    # def get_new_goal(self, goal_x, goal_y, reason, odom_state_sequence):
    def get_index(self, odom_state_sequence):

        # state_sequence = []
        # with open('state_squence', 'r') as f:
        #     state_sequence = f.read()
        # goal = [goal_x, goal_y]
        # response_dict = '{"goal_x": x, "goal_y": y}'

        # prompt_string = '''
        # Given the context of reinforcement learning and hindsight experience replay, generate a substitute goal that guides the robot toward learning the optimal behavior pattern for faster convergence.

        # ### Instructions:
        # 1. Identify a goal point from the provided trajectory that is located on the longest, straightest, and safest segment of the path.
        # 2. Prefer goal points that are farthest from the starting position and avoid areas near potential collisions (ensure min_laser > 0.4). The selected goal should encourage safe and smooth navigation.
        # 3. Prioritize long, stable segments of the trajectory over shorter or more curved ones to promote learning a more efficient behavior.
        # 4. The robot will navigate to the selected goal point, and any remaining trajectory points will be pruned.

        # ### Details:
        # - The original goal in this trajectory was at coordinates (goal_x, goal_y): {}.
        # - The robot did not achieve this goal because {}.
        # - The trajectory’s first 100 steps (fewer if the trajectory is shorter) are structured as: [robot_odom_x, robot_odom_y, distance_to_goal, min_laser_distance, theta_between_goal, linear_velocity.x, angular_velocity.z]: {}.

        # ### Expected Output:
        # Please provide a goal coordinate that maximizes cumulative reward while considering:
        # - Favoring the longest straight and safe segments of the trajectory.
        # - Ensuring safety with min_laser > 0.4.
        # - The response should be strictly structured as: {}, please only return in this format without extra information.
        # '''.format(goal, reason, odom_state_sequence, response_dict)

        response_dict = '{"start_index": i, "end_index": j}'
        prompt_string = '''
            Given the context of reinforcement learning and hindsight experience replay, your task is to identify the longest safe trajectory segment from the provided data and return its starting and ending indices.

            ### Instructions:
            1. Analyze the trajectory data to find the longest contiguous segment where the robot maintains a safe distance from obstacles (min_laser > 0.4).
            2. Ensure that the selected segment is as long as possible while adhering to the safety constraint.
            3. The selected segment should represent the robot's best performance in terms of safety and continuity.
            4. The segment must be at least 10 steps long to be considered valid.
            5. The starting index must be greater than 0, and the ending index must be less than the length of the trajectory.
            6. Return the starting and ending indices of the identified segment within the original trajectory.

            ### Details:
            - The trajectory data is structured as a list of steps, where each step contains: [robot_odom_x, robot_odom_y, min_laser_distance].
            - The safety constraint is defined as min_laser > 0.4, indicating a safe distance from potential collisions.
            - An example of a valid trajectory segment: If the trajectory is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and the longest safe segment is [4, 5, 6, 7, 8], then the start_index is 3 and the end_index is 7.
            - The trajectory’s first 100 steps (or fewer if the trajectory is shorter) are provided: {}.

            ### Expected Output:
            Provide the starting and ending indices of the longest safe trajectory segment as a JSON object.
            The response should be strictly structured as: {}, please only return in this format without extra information.
        '''.format(odom_state_sequence, response_dict)





        # response = self.client.chat.completions.create(
        #     model=self.deployment_name,
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": prompt_string}
        #     ]
        # )


        # prompt_string = '''
        #     Given the context of reinforcement learning and hindsight experience replay, provide a new goal point that maximizes the reward for the given trajectory. The goal should be achievable given the robot’s final position and the provided trajectory.

        #     Instructions:
        #     1. The goal should be feasible based on the robot’s final position and should lie within the environment's bounds.
        #     2. The objective is to reuse this trajectory and generate the highest possible reward.
        #     3. The goal should not very close to the collision point, otherwise it will lead the robot to make collisions.

        #     Details:
        #     - The original goal in this trajectory is (goal_x, goal_y): {}.
        #     - The robot did not achieve this goal, due to {}.
        #     - The last 50 sequence of steps ([robot_odom_x, robot_odom_y, distance to goal, min_laser_distance, theta between goal, linear_velocity.x, angular_velocity.z]) in this trajectory are: {}.

        #     Please only return me a goal coordinate and The response should strictly look like: '{}'.
        #     '''.format(goal, reason, odom_state_sequence, response_dict)

        # response_dict = "{'state_sequence': [state1, state2, state3, ...], 'next_state_squence':[next_state1, next_state2, next_state3, ...], 'odom_x_sequence': [odom_x1, odom_x2, odom_x3, ...], 'odom_y_sequence': [odom_y1, odom_y2, odom_y3, ...], 'angle_sequence':[angle1, angle2, angle3, ...], 'goal_x': x, 'goal_y': y}"
        # prompt_string = '''
        #     Given the historical trajectory of a robot navigating towards a goal, generate the next sequence of states that would optimize reward while strictly avoiding collisions.

        #     ### Task and Instructions:
        #     1. Generate a trajectory that maintains a safe distance from obstacles and efficiently reaches the target.
        #     2. Ensure the trajectory avoids collisions with obstacles and adheres to the safe distance threshold.
        #     3. The trajectory should be smooth, maintaining a safe distance of at least 0.35 units from obstacles at all times.
        #     4. Follow patterns learned from previous trajectories to optimize the path and improve the RL training process.

        #     ### Historical Trajectory:
        #     - The last 50 states (or fewer if it's a short trajectory) are: {}
                    
        #     ### Task Details:
        #     - Current Target Position (goal_x, goal_y): {}
        #     - The robot did not achieve the goal due to: {}
        #     - The robot's odom sequence (odom_x, odom_y): {}
        #     - The robot's angle sequence: {}
        #     - Safe Distance Threshold: 0.35

        #     ### Expected Output:
        #     - A series of states representing the robot’s trajectory.
        #     - The structure of each state should be as follows: [laser_distances(20 sectors), distance_to_goal, theta_between_goal, linear_velocity.x, angular_velocity.z], with each state having a dimension of 24.
        #     - The next state is the state after taking action, and the sequence should have the same length as the state sequence.
        #     - nsure all states maintain a minimum distance of 0.35 units from obstacles.
        #     - All floating-point values should be rounded to 3 decimal places.
        #     - The response should be strictly structured as: {}, please only return in this format without extra information.
        #     '''.format(state_sequence, goal, reason, odom_sequence, angle_sequence, response_dict)



        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                { "role": "system", "content": "You are a helpful assistant specialized in sample generation for reinforcement learning." },
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
        print(content)
        # print('new goal',response)
        # match = re.search(r'\{.*\}', content.strip(), re.DOTALL)
        # if match:
        #     json_str = match.group()  # 提取到的 JSON 字符串
        #     # 解析 JSON 字符串
        #     response_dict = json.loads(json_str)
        #     if response_dict is None:
        #         return goal_x, goal_y
        # else:
        #     print("No JSON object found")
        #     return goal_x, goal_y
        
        match = re.search(r'\{.*\}', content.strip(), re.DOTALL)
        if match:
            json_str = match.group()  # 提取到的 JSON 字符串
            # 解析 JSON 字符串
            response_dict = json.loads(json_str)
            if response_dict is None:
                return 0, len(odom_state_sequence)-1
        else:
            print("No JSON object found")
            return 0, len(odom_state_sequence)-1



        # # 提取 goal_x 和 goal_y
        # state_sequence = np.array(response_dict.get('state_sequence')).tolist()
        # next_state_sequence = np.array(response_dict.get('next_state_sequence')).tolist()
        # odom_x_sequence = np.array(response_dict.get('odom_x_sequence')).tolist()
        # odom_y_sequence = np.array(response_dict.get('odom_y_sequence')).tolist()
        # angle_sequence = np.array(response_dict.get('angle_sequence')).tolist()
        # goal_x = response_dict.get('goal_x')
        # goal_y = response_dict.get('goal_y')
        start_index = response_dict.get('start_index')
        end_index = response_dict.get('end_index')

        # print('new goal', goal_x, goal_y)
        # print('state sequence', state_sequence)
        # print('next state sequence', next_state_sequence)
        # print('odom x sequence', odom_x_sequence)
        # print('odom y sequence', odom_y_sequence)
        # print('angle sequence', angle_sequence)
   

        # return state_sequence, next_state_sequence, odom_x_sequence, odom_y_sequence, angle_sequence, goal_x, goal_y
        # return goal_x, goal_y
        return start_index, end_index
    


# api_base = "https://ai-azureaiwestusengine881797519629.openai.azure.com/"
# api_key = '4d44030e3b2b4f978a44cccc49b464f1'
# deployment_name = 'gpt-4'
# api_version = '2023-03-15-preview'

# image_scorer = ImageScorer(api_base, api_key, deployment_name, api_version)
# map_index = 1950
# score = image_scorer.get_score_for_images(map_index)
