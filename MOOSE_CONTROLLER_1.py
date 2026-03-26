"""MOOSE_CONTROLLER_1 controller."""


print("All set")

from controller import Robot
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
import cv2

robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0  # seconds

#math

import math

def deg2rad(degrees):
    return degrees * math.pi / 180


def rad2deg(radians):
    return radians * 180 / math.pi

# ---- Sensors ----
Gyro = robot.getDevice("gyro")
Gyro.enable(timestep)


GPS = robot.getDevice("gps(1)")
GPS.enable(timestep)

inertial_unit = robot.getDevice("inertial unit")
inertial_unit.enable(timestep)

accelerometer = robot.getDevice("accelerometer")
accelerometer.enable(timestep)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)
lidar.enablePointCloud()

camera = robot.getDevice("camera")
camera.enable(timestep)

# ---- Encoder sensors ----
enc_rm1 = robot.getDevice("rm1")
enc_rm2 = robot.getDevice("rm2")
enc_rm3 = robot.getDevice("rm3")
enc_rm4 = robot.getDevice("rm4")

enc_lm1 = robot.getDevice("lm1")
enc_lm2 = robot.getDevice("lm2")
enc_lm3 = robot.getDevice("lm3")
enc_lm4 = robot.getDevice("lm4")

encoders = [
    enc_rm1, enc_rm2, enc_rm3, enc_rm4,
    enc_lm1, enc_lm2, enc_lm3, enc_lm4
]

for e in encoders:
    e.enable(timestep)

# ---- Motors ----
motor_l1 = robot.getDevice("left motor 1")
motor_l2 = robot.getDevice("left motor 2")
motor_l3 = robot.getDevice("left motor 3")
motor_l4 = robot.getDevice("left motor 4")

motor_r1 = robot.getDevice("right motor 1")
motor_r2 = robot.getDevice("right motor 2")
motor_r3 = robot.getDevice("right motor 3")
motor_r4 = robot.getDevice("right motor 4")

motors = [
    motor_l1, motor_l2, motor_l3, motor_l4,
    motor_r1, motor_r2, motor_r3, motor_r4
]

# Velocity control mode
for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# ---- Motor control function ----
def set_motor_velocities(l1, l2, l3, l4, r1, r2, r3, r4):
    motor_l1.setVelocity(l1)
    motor_l2.setVelocity(l2)
    motor_l3.setVelocity(l3)
    motor_l4.setVelocity(l4)

    motor_r1.setVelocity(r1)
    motor_r2.setVelocity(r2)
    motor_r3.setVelocity(r3)
    motor_r4.setVelocity(r4)


def rotate_clockwise(speed):
    set_motor_velocities(speed,speed,speed,speed,-speed,-speed,-speed,-speed)

def forward(speed):
    set_motor_velocities(speed,speed,speed,speed,speed,speed,speed,speed)

# ---- Encoder reading ----


def get_encoder_values():
    global initial_pose

    values = np.array([e.getValue() for e in encoders], dtype=float)    
    #print(values)
    return values 

def read_gps(gps=GPS):
    values = gps.getValues()  # [x, y, z]
    x = values[0]
    y = values[1]
    z = values[2]
    return x, y, z

def get_inertial_data(inertial_unit=inertial_unit):
    values = inertial_unit.getRollPitchYaw()
    
    roll = values[0]
    pitch = values[1]
    yaw = values[2]
    
    return roll,pitch,yaw


def display_lidar_map(ranges, angles,
                      x_robot, y_robot, yaw,
                      x_target, y_target,
                      size=600, scale=10):

    img = np.zeros((size, size, 3), dtype=np.uint8)

    cx = size // 2
    cy = size // 2

    # draw lidar points
    for r, a in zip(ranges, angles):
        if np.isinf(r) or np.isnan(r):
            continue

        x = r * math.cos(a)
        y = r * math.sin(a)

        px = int(cx + x * scale)
        py = int(cy - y * scale)

        if 0 <= px < size and 0 <= py < size:
            img[py, px] = (255, 0, 0)

    # robot position
    cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)

    # --- convert GPS target to robot frame ---
    dx = x_target - x_robot
    dy = y_target - y_robot

    x_local = dx * math.cos(yaw) + dy * math.sin(yaw)
    y_local = -dx * math.sin(yaw) + dy * math.cos(yaw)

    # convert to pixels
    tx = int(cx + x_local * scale)
    ty = int(cy - y_local * scale)

    #if 0 <= tx < size and 0 <= ty < size:
    #    cv2.circle(img, (tx, ty), 6, (0, 0, 255), -1)

    cv2.imshow("LiDAR", img)
    cv2.waitKey(1)

# ---- Odometry ---

wheel_radius = 0.319
track_width = 3.4

x = 0
y = 0
theta = 0

prev_enc = get_encoder_values()
prev_values = None

def update_odometry():
    global x, y, theta, prev_values

    values = get_encoder_values()

    if prev_values is None:
        prev_values = values
        return x, y, theta

    dphi = [(v - pv) for v, pv in zip(values, prev_values)]
    prev_values = values

    dphi_r = statistics.mean(dphi[0:4])
    dphi_l = statistics.mean(dphi[4:8])

    dSr = wheel_radius * dphi_r
    dSl = wheel_radius * dphi_l

    dS = (dSl + dSr) / 2
    dtheta = (dSr - dSl) / track_width

    x += dS * np.cos(theta + dtheta / 2)
    y += dS * np.sin(theta + dtheta / 2)
    imu_theta = inertial_unit.getRollPitchYaw()[2]

    theta = theta + dtheta
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    return x, y, theta


import numpy as np



def rotation_direction(theta, phi):
    delta = (phi - theta + math.pi) % (2 * math.pi) - math.pi

    if delta > 0:
        return 1
    elif delta < 0:
        return -1
    else:
        return 0

prev_dyaw = 0.0
prev_dist = 0.0

def turn_to_pose(yaw_target):
    global prev_dyaw
    global v_robot
    global v_damped

    roll, pitch, yaw = get_inertial_data()
    direction = rotation_direction(yaw,yaw_target)
    dyaw = yaw_target - yaw

    kp = 5.0
    kd = 0.3
    max_speed = 12
    tolerance = 0.005

    derivative = dyaw - prev_dyaw
    prev_dyaw = dyaw

    speed = kp * abs(dyaw) + kd * abs(derivative)
    speed = np.clip(speed, -max_speed, max_speed)

    if speed - v_robot > 1:
        speed = v_damped*1.03
        v_damped = speed

    v_robot = speed

    if direction == -1:
        rotate_clockwise(speed)
    elif direction == 1:
        rotate_clockwise(-speed)
    elif direction == 0:
        set_motor_velocities(0,0,0,0,0,0,0,0)

    if abs(dyaw) < tolerance:
        set_motor_velocities(0,0,0,0,0,0,0,0)
        v_damped = 0.1
        return True

    return False

sequence = 0
v_robot = 0
v_damped = 0.1

def nav_2_pose(x_target, y_target, theta_target):
    global sequence
    global prev_dist
    global v_robot
    global v_damped

    pos_tolerance = 0.05

    x, y, z = read_gps()
    roll, pitch, yaw = get_inertial_data()
    print(x,y,yaw)
    dx = x_target - x
    dy = y_target - y
    distance = math.sqrt(dx**2 + dy**2)

    if sequence == 0:
        heading_target = math.atan2(dy, dx)
        A = turn_to_pose(heading_target)
        if A:
            sequence +=1

    elif sequence == 1:
        kp = 3
        kd = 0.5
        derivative = distance - prev_dist
        prev_dist = distance

        v = kp*distance + kd*derivative
        v = min(v,10)

        if v - v_robot > 2:
            v = v_damped*1.03
            v_damped = v

        forward(v)
        v_robot = v

        if distance <= pos_tolerance:
            forward(0)
            sequence += 1
            v_damped = 0.1

    elif sequence == 2:
        A = turn_to_pose(theta_target)
        if A:
            forward(0)
            sequence = 0
            return True

    return False


def get_lidar_data():
    global lidar
    
    values = lidar.getRangeImage()
    fov = lidar.getFov()
    resolution = lidar.getHorizontalResolution()

    angle_step = fov / resolution
    start_angle = -fov / 2

    ranges = []
    angles = []

    for i, r in enumerate(values):
        angle = start_angle + i * angle_step
        ranges.append(r)
        angles.append(angle)

    return ranges, angles

while robot.step(timestep) != -1:
    x_target = 5
    y_target = 5
    yaw_target = deg2rad(-60)
    #plot_map(x_target,y_target,yaw_target)
    x, y, z = read_gps()
    roll, pitch, yaw = get_inertial_data()
    ranges, angles = get_lidar_data()
    display_lidar_map(ranges, angles, x,y,yaw,x_target,y_target)

    A = nav_2_pose(x_target,y_target,yaw_target,)
    if A:
        forward(0)
        print("TARGET REACHED")
        break