"""
File for defining hardware sensors
"""
from enum import Enum
import numpy as np

class SensorState(Enum):
    """An enumeration of different sensor states."""
    ACTIVE = 1
    INACTIVE = 2
    FAULT = 3


class Encoder:
    def __init__(self, pb_client, id, bot):
        self.pb_client = pb_client
        self.state = SensorState.ACTIVE
        self.id = id
        self.bot = bot
    
    def get_motor_position_info(self):
        pos, vel, react, torque = self.pb_client.getJointState(self.bot, self.id)
        return pos, vel, react, torque
    

class Accelerometer:
    def __init__(self, pb_client, bot):
        self.pb_client = pb_client
        self.state = SensorState.ACTIVE
        self.bot = bot
    
    def read_sensor(self):
        output = self.pb_client.getBasePositionAndOrientation(self.bot)
        return output
    
    def calculate_velocity(self):
        output =self.pb_client.getBaseVelocity(self.bot)
        return output
    
    def _get_obs(self):
        output = self.pb_client.getBasePositionAndOrientation(self.bot)
        return np.array(output).astype('float64')
