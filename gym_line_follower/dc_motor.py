class DCMotor:
    """
    Simple DC motor model.

    Motor specifications exist in bot_config.json.
    """

    def __init__(self, nominal_voltage, no_load_speed, stall_torque, stall_current, state=1):
        """
        Create a motor with parameters:
        :param no_load_speed: [rad/s]
        :param nominal_voltage: [V]
        :param stall_current: [A]
        :param stall_torque: [Nm]
        """
        self.state = state
        self.motor_constant, self.winding_resistance = self.get_motor_parameters(nominal_voltage, no_load_speed, stall_torque, stall_current)

    def get_torque(self, input_voltage, w):
        """
        Calculate instant torque from input voltage and rotation speed.
        :param input_voltage: [V]
        :param w: [rad/s]
        :return: shaft torque [Nm]
        """
        return (input_voltage - w * self.motor_constant) / self.winding_resistance * self.motor_constant

    def get_efficiency(self, input_voltage, w, output_torque):
        """
        Calculate the power efficiency of the DC Motor from input voltage, rotation speed, and output torque.
        :param input_voltage: [V]
        :param w: [rad/s]
        :param output_torque: [Nm]
        :return: power efficiency
        """
        power_in = input_voltage * (output_torque / self.motor_constant)
        power_out = output_torque * w
        return power_out / power_in

    def get_torque_and_efficiency(self, input_voltage, w):
        """
        Calculates the torque and efficiency of the DC Motor given an input voltage and rotation speed.
        :param input_voltage: [V]
        :param w: [rad/s]
        :return: tuple of: output_torque, power efficiency
        """
        output_torque = self.get_torque(input_voltage, w)
        efficiency = self.get_efficiency(input_voltage, w, output_torque)
        return output_torque, efficiency


    @staticmethod
    def get_motor_parameters(nominal_voltage,  no_load_speed, stall_torque, stall_current):
        """
        Calculate motor constant and resistance from parameters. Assumes an ideal motor where
        the motor constant K = K_t = K_e, where K_t = torque constant [Nm/A] and K_e = back electromotive
        force constant [V/(rad/s)].
        :param nominal_voltage: [V]
        :param no_load_speed: [rad/s]
        :param stall_torque: [Nm]
        :param stall_current: [A]
        :return: tuple of: motor constant, resistance
        """
        motor_constant = stall_torque / stall_current
        winding_resistance = nominal_voltage / stall_current
        #friction = motor_constant * (nominal_voltage - motor_constant * no_load_speed) / (winding_resistance * no_load_speed)
        return motor_constant, winding_resistance


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    v = np.linspace(0, 6, 100)
    motor = DCMotor(6., 105., 0.057)
    T = [motor.get_torque(vs, 105.) for vs in v]
    plt.plot(v, T)
    plt.grid()
    plt.show()
