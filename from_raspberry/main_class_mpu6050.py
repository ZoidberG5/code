import time
from mpu6050_sensor.mpu6050_OOP import MPU6050Sensor

if __name__ == "__main__":
    # Create an instance of the MPU6050Sensor
    mpu_manager = MPU6050Sensor()
    while True:
        # Get filtered acceleration data
        filtered_x, filtered_y, filtered_z = mpu_manager.get_filtered_acceleration()
        print(f"Filtered Acceleration -> X: {filtered_x}, Y: {filtered_y}, Z: {filtered_z}")

        roll, pitch = mpu_manager.get_filtered_roll_pitch()
        print(f"Filtered roll and pitch -> roll: {roll}, pitch: {pitch}")

        roll, pitch = mpu_manager.get_avg_filtered_roll_pitch(count = 20)
        print(f"avarage filtered roll and pitch -> roll: {roll}, pitch: {pitch}")
        time.sleep(4)