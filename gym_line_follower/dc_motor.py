class DCMotor:
    """
    Simple DC motor model.

    Motor specifications exist in bot_config.json.
    """

    def __init__(self, nominal_voltage, no_load_speed, free_current, stall_current, state=1):
        """
        Create a motor with parameters:
        :param no_load_speed: [rad/s]
        :param nominal_voltage: [V]
        :param free_current: [A]
        :param stall_current: [A]
        """
        self.state = state
        self.nominal_voltage = nominal_voltage
        self.no_load_speed = no_load_speed
        self.free_current = free_current
        self.stall_current = stall_current

        # calculate motor constant and electrical resistance
        self.motor_constant, self.winding_resistance = self.get_motor_parameters()

    def get_torque_and_efficiency(self, input_voltage, w):
        """
        Calculates the torque and efficiency of the DC Motor given an input voltage and rotation speed.
        :param input_voltage: [V]
        :param w: [rad/s]
        :param free_current: [A]
        :return: tuple of: output_torque, power efficiency
        """
        output_torque, load_current = self.get_torque_and_load_current(input_voltage, w)
        power_out, power_in = self.get_power_out_and_power_in(input_voltage, load_current, output_torque, w)
        efficiency = power_out / power_in
        return output_torque, efficiency

    def get_torque_and_load_current(self, input_voltage, w):
        """
        Calculate instant torque from input voltage and rotation speed.
        :param input_voltage: [V]
        :param w: [rad/s]
        :return: shaft torque [Nm] and current [A]
        """
        load_current = (input_voltage - w * self.motor_constant) / self.winding_resistance
        torque = load_current * self.motor_constant
        return torque, load_current

    def get_power_out_and_power_in(self, input_voltage, load_current, output_torque, w):
        """
        Calculate the power efficiency of the DC Motor from input voltage, rotation speed, and output torque.
        :param input_voltage: [V]
        :param current [A]
        :param output_torque: [Nm]
        :param w: [rad/s]
        :return: power efficiency
        """
        # P_in = I * V
        power_in = (load_current + self.free_current) * input_voltage

        # P_out = T * w
        power_out = output_torque * w

        return power_out, power_in

    ### outdated get_torque func ###
    def get_torque(self, input_voltage, w):
        """
        Calculate instant torque from input voltage and rotation speed.
        :param input_voltage: [V]
        :param w: [rad/s]
        :return: shaft torque [Nm]
        """
        return (input_voltage - w * self.motor_constant) / self.winding_resistance * self.motor_constant
    ### outdated get_torque func ###

    def get_motor_curve_data(self, input_voltage, w):
        """
        Calculates the torque, load current, power out and power in of the DC Motor 
        given an input voltage and rotation speed. NOTE: Used for plotting purposes!
        :param input_voltage: [V]
        :param w: [rad/s]
        :return: tuple of: output_torque, power_out, power_in, load_current
        """
        output_torque, load_current = self.get_torque_and_load_current(input_voltage, w)
        power_out, power_in = self.get_power_out_and_power_in(input_voltage, load_current, output_torque, w)
        return output_torque, power_out, power_in, load_current


    def get_motor_parameters(self):
        """
        Calculate motor constant and resistance from parameters. Assumes an ideal motor where
        the motor constant K = K_t = K_e, where K_t = torque constant [Nm/A] and K_e = back electromotive
        force constant [V/(rad/s)].
        :return: tuple of: motor constant, resistance
        """
        winding_resistance = self.nominal_voltage / self.stall_current
        motor_constant = (self.nominal_voltage - (self.free_current * winding_resistance)) / self.no_load_speed
        return motor_constant, winding_resistance


if __name__ == '__main__':
    """
    Creates a DCMotor and plots the motor curve data.
    """
    # Dependencies
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from randomizer_dict import RandomizerDict
    
    # Open the bot_config file
    local_dir = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(local_dir, "bot_config.json")
    config = RandomizerDict(json.load(open(config_path, "r")))

    # Get motor specs from config file and create DC motor
    nom_volt = config["motor_nominal_voltage"]
    stall_current = config["motor_stall_current"]
    stall_torque = config["motor_stall_torque"]
    no_load_speed = config["motor_no_load_speed"]
    free_current = config["motor_free_current"]
    front_left_motor = DCMotor(nom_volt, no_load_speed, free_current, stall_current, state=1.0)

    # get torque, power out, power in, and load current over a range of angular velocities
    input_voltages = np.full(2055, 12.0) # volts
    angular_velocities = np.linspace(0.0, 2054,2055) # rad/s
    torques, power_out, power_in, currents = front_left_motor.get_motor_curve_data(input_voltages, angular_velocities)
    
    # calculate efficiency
    efficiencies = (power_out / power_in) * 100

    # convert angular velocities from rad/s to RPM
    angular_velocities *= 60 / (2*np.pi)

    # create fig and axes
    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.75)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.2))
    
    # plot
    ax1.plot(angular_velocities, currents, label="Current (A)", color="b", linestyle="dotted")
    ax1.plot(angular_velocities, power_out, label="Power Out (W)", color="g")
    ax2.plot(angular_velocities, torques, label="Torque (Nm)", color="y")
    ax3.plot(angular_velocities, efficiencies, label="Efficiency (%)", color="r", linestyle="dashed")
    
    # set axis labels and title
    ax1.set_ylabel("Current (A), Power Out (W)")
    ax2.set_ylabel("Torque (Nm)")
    ax3.set_ylabel("Efficiency (%)")
    ax1.set_xlabel("Angular Velocity (RPM)")
    ax1.set_title("RS-550 Motor Curve Data at Nominal Voltage = 12 V")

    # set x and y axis ticks
    ax3.set_yticks(np.arange(0, 110, step=10))
    plt.xticks(np.arange(0, 22000, step=2000))
    ax1.set_ylim(ymin=0)
    ax2.set_ylim(ymin=0)
    ax3.set_ylim(ymin=0)
    ax1.set_xlim(xmin=0)
    ax2.set_xlim(xmin=0)
    ax3.set_xlim(xmin=0)
    
    # create legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc=0)
    ax1.grid()

    # motor curve data
    peak_power = np.round(power_out.max(), 3)
    peak_power_ang_vel = np.round(angular_velocities[np.argmax(power_out)], 3)
    peak_eff = np.round(efficiencies.max(), 3)
    peak_eff_output_torque = np.round(torques[np.argmax(efficiencies)], 3)
    peak_eff_ang_vel = np.round(angular_velocities[np.argmax(efficiencies)], 3)

    print("Peak Power Out ", peak_power, "W at ", peak_power_ang_vel, " RPM")
    print("Peak Efficiency ", peak_eff, "% ", 
          "at Output Torque ", peak_eff_output_torque, 
          " and Angular Velocity ", peak_eff_ang_vel, " RPM")
    plt.show()
