import scipy.spatial.transform
import numpy as np
from animate_function import QuadPlotter
import csv
import pandas as pd
import mlmodel
import torch

def quat_mult(q, p):
    # q * p
    # p,q = [w x y z]
    return np.array(
        [
            p[0] * q[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3],
            q[1] * p[0] + q[0] * p[1] + q[2] * p[3] - q[3] * p[2],
            q[2] * p[0] + q[0] * p[2] + q[3] * p[1] - q[1] * p[3],
            q[3] * p[0] + q[0] * p[3] + q[1] * p[2] - q[2] * p[1],
        ]
    )
    
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_from_vectors(v_from, v_to):
    v_from = normalized(v_from)
    v_to = normalized(v_to)
    v_mid = normalized(v_from + v_to)
    q = np.array([np.dot(v_from, v_mid), *np.cross(v_from, v_mid)])
    return q

def normalized(v):
    norm = np.linalg.norm(v)
    return v / norm

NO_STATES = 13
IDX_POS_X = 0
IDX_POS_Y = 1
IDX_POS_Z = 2
IDX_VEL_X = 3
IDX_VEL_Y = 4
IDX_VEL_Z = 5
IDX_QUAT_W = 6
IDX_QUAT_X = 7
IDX_QUAT_Y = 8
IDX_QUAT_Z = 9
IDX_OMEGA_X = 10
IDX_OMEGA_Y = 11
IDX_OMEGA_Z = 12

class Robot:
    
    '''
    frames:
        B - body frame
        I - inertial frame
    states:
        p_I - position of the robot in the inertial frame (state[0], state[1], state[2])
        v_I - velocity of the robot in the inertial frame (state[3], state[4], state[5])
        q - orientation of the robot (w=state[6], x=state[7], y=state[8], z=state[9])
        omega - angular velocity of the robot (state[10], state[11], state[12])
    inputs:
        omega_1, omega_2, omega_3, omega_4 - angular velocities of the motors
    '''
    def __init__(self):
        self.m = 1.0 # mass of the robot
        self.arm_length = 0.25 # length of the quadcopter arm (motor to center)
        self.height = 0.05 # height of the quadcopter
        self.body_frame = np.array([(self.arm_length, 0, 0, 1),
                                    (0, self.arm_length, 0, 1),
                                    (-self.arm_length, 0, 0, 1),
                                    (0, -self.arm_length, 0, 1),
                                    (0, 0, 0, 1),
                                    (0, 0, self.height, 1)])

        self.J = 0.025 * np.eye(3) # [kg m^2]
        self.J_inv = np.linalg.inv(self.J)
        self.constant_thrust = 10e-4
        self.constant_drag = 10e-6
        self.omega_motors = np.array([0.0, 0.0, 0.0, 0.0])
        self.state = self.reset_state_and_input(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        self.time = 0.0

        # NF
        self.a = np.zeros((3,1))
        self.lamb = 1.0
        self.P = np.eye(3) * 0.5
        self.R = np.eye(3) * 0.5
        self.Q = np.eye(3) * 0.1

        #Logging
        self.traj = []
        self.p_d_I = np.array([0.0, 0.0, 0.0])
        self.v_d_I = np.array([0.0, 0.0, 0.0])
        self.fa = np.array([0.0, 0.0, 0.0])
        self.R_B_to_I = np.eye(3)
        self.T_sp = np.array([0.0, 0.0, 0.0, 0.0])
        self.q_sp = np.array([0.0, 0.0, 0.0, 0.0])


    def reset_state_and_input(self, init_xyz, init_quat_wxyz):
        state0 = np.zeros(NO_STATES)
        state0[IDX_POS_X:IDX_POS_Z+1] = init_xyz
        state0[IDX_VEL_X:IDX_VEL_Z+1] = np.array([0.0, 0.0, 0.0])
        state0[IDX_QUAT_W:IDX_QUAT_Z+1] = init_quat_wxyz
        state0[IDX_OMEGA_X:IDX_OMEGA_Z+1] = np.array([0.0, 0.0, 0.0])
        return state0

    def update(self, omegas_motor, dt):
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]
        R = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        thrust = self.constant_thrust * np.sum(omegas_motor**2)
        f_b = np.array([0, 0, thrust])
        
        tau_x = self.constant_thrust * (omegas_motor[3]**2 - omegas_motor[1]**2) * 2 * self.arm_length
        tau_y = self.constant_thrust * (omegas_motor[2]**2 - omegas_motor[0]**2) * 2 * self.arm_length
        tau_z = self.constant_drag * (omegas_motor[0]**2 - omegas_motor[1]**2 + omegas_motor[2]**2 - omegas_motor[3]**2)
        tau_b = np.array([tau_x, tau_y, tau_z])

        v_dot = 1 / self.m * R @ f_b + np.array([0, 0, -9.81])
        omega_dot = self.J_inv @ (np.cross(self.J @ omega, omega) + tau_b)
        q_dot = 1 / 2 * quat_mult(q, [0, *omega])
        p_dot = v_I
        
        x_dot = np.concatenate([p_dot, v_dot, q_dot, omega_dot])
        self.state += x_dot * dt
        self.state[IDX_QUAT_W:IDX_QUAT_Z+1] /= np.linalg.norm(self.state[IDX_QUAT_W:IDX_QUAT_Z+1]) # Re-normalize quaternion.
        self.time += dt

        #Logging
        self.traj.append(self.get_traj_NF_train())
        R = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    def get_traj_NF_train(self):
        t = self.time

        # Vectors: convert to list and then to string.
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        p_I_str = str(p_I.tolist())

        p_d_I = self.p_d_I
        p_d_I_str = str(p_d_I.tolist())  # assuming p_d_I is array-like

        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        v_I_str = str(v_I.tolist())

        v_d_I = self.v_d_I
        v_d_I_str = str(v_d_I.tolist())

        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        q_str = str(q.tolist())

        # Rotation matrix.
        R_B_to_I = self.R_B_to_I
        if isinstance(R_B_to_I, (list, np.ndarray)):
            R_B_to_I_str = str(np.array(R_B_to_I).tolist())
        else:
            R_B_to_I_str = str(R_B_to_I)

        omega = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]
        omega_str = str(omega.tolist())

        # For scalar values, wrap them in a list so they appear with [].
        T_sp = self.T_sp
        T_sp_str = str([T_sp])

        q_sp = self.q_sp
        q_sp_str = str(q_sp.tolist())  # assuming q_sp is array-like

        hover_throttle = 0.5
        hover_throttle_str = str([hover_throttle])

        fa = self.fa
        # If fa is a NumPy array, convert it to list; otherwise, wrap in list.
        if isinstance(fa, np.ndarray):
            fa_str = str(fa.tolist())
        else:
            fa_str = str([fa])

        # Compute pwm as a NumPy array and convert to list-string.
        pwm = self.omega_motors # Techinically not pwm but its fine for now.
        pwm_str = str(pwm.tolist())

        # Return the trajectory list (order: t, p, p_d, v, v_d, q, R, omega, T_sp, q_sp, hover_throttle, fa, pwm)
        return [t, p_I_str, p_d_I_str, v_I_str, v_d_I_str, q_str,
                R_B_to_I_str, omega_str, T_sp_str, q_sp_str,
                hover_throttle_str, fa_str, pwm_str]

    
    def control(self, p_d_I, v_d_I, model):
        self.p_d_I = p_d_I
        self.v_d_I = v_d_I

        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega_b = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]

        # Position controller.
        k_p = 1.0
        k_d = 10.0
        v_r = v_d_I - k_p * (p_I - p_d_I)
        a = -k_d * (v_I - v_r) + np.array([0, 0, 9.81])
        f = self.m * a

        # Wind disturbance.
        F0 = 8
        # w_wind = np.pi/4
        # phi = 0.5
        #f_wind = F0 * np.array([np.sin(w_wind * self.time+ phi), 0, 0])
        f_wind = F0 * np.array([1, 0, 0])# constant wind
        f += f_wind

        #-----------------Adaptive Control NF-----------------
        # Compute Phi from partial state
        X = torch.from_numpy(np.concatenate([v_I, q, self.omega_motors])).flatten()
        phi_val  = model.phi(X).detach().numpy().reshape(-1)
        #Phi  = np.diag(phi_val)
        Phi  = np.eye(3)
        #print("|Phi|  ",np.linalg.norm(Phi))
        Phi_T = np.transpose(Phi) 

        #print("Phi, Phi_T shape: ",Phi.shape, Phi_T.shape)
        #----------------------Update a------------------------
        s = (v_I - v_r).reshape(3, 1) 
 
        # ===[Regularization Term]===
        a_dot_reg = - self.lamb * self.a 
        #print("a_dot_reg shape: ", a_dot_reg.shape)
        # ===[Prediction Error Term]===
        error = Phi @ self.a - f_wind.reshape(3,1) 
        # print("R shape: ",self.R.shape)
        #print("error shape: ",error.shape)
        a_dot_pred = - self.P @ Phi_T@ np.linalg.pinv(self.R) @ error
        #print("a_dot_pred shape: ",a_dot_pred.shape)
        #===[Tracking Error Term]===
        a_dot_track_err =self.P @ Phi_T @ s
        #print("a_dot_track_err shape: ",a_dot_track_err.shape)
        # [Finally, update a]
        a_dot = a_dot_pred + a_dot_reg + a_dot_track_err
        #print("a_dot shape: ",a_dot.shape)
        self.a += a_dot * dt
        #print("a: ",self.a)
        #-----------------------Update P----------------------
        P_dot = -2 * self.lamb * self.P + self.Q - self.P @ Phi_T @ np.linalg.pinv(self.R) @ Phi @ self.P 
        self.P += P_dot * dt
        #print("u_NF  ",-Phi@self.a)
        f+= (- Phi @ self.a).flatten()
        print("NF PD |s|  ",np.linalg.norm(s))
        # if(self.time > 2):
        #     #print("u_NF  ",-Phi@self.a)
        #     #print("f_wind  ",f_wind)
        #     f+= (- Phi @ self.a).flatten()
        #     print("NF PD |s|  ",np.linalg.norm(s))
        # else:
        #     print("PD |s|  ",np.linalg.norm(s))

        R_B_to_I = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        self.R_B_to_I = R_B_to_I
        R_I_to_B = R_B_to_I.T
        # Wind in body frame.
        self.fa = R_I_to_B @ f_wind
        f_b = R_I_to_B @ f
        thrust = np.max([0, f_b[2]])
        self.T_sp = thrust

        # Attitude controller.
        q_ref = quaternion_from_vectors(np.array([0, 0, 1]), normalized(f))
        self.q_sp = q_ref
        q_err = quat_mult(quat_conjugate(q_ref), q) # error from Body to Reference.
        if q_err[0] < 0:
            q_err = -q_err
        k_q = 20.0
        k_omega = 100.0
        omega_ref = - k_q * 2 * q_err[1:]
        alpha = - k_omega * (omega_b - omega_ref)
        tau = self.J @ alpha + np.cross(omega_b, self.J @ omega_b)
        
        # Compute the motor speeds.
        B = np.array([
            [self.constant_thrust, self.constant_thrust, self.constant_thrust, self.constant_thrust],
            [0, -self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust],
            [-self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust, 0],
            [self.constant_drag, -self.constant_drag, self.constant_drag, -self.constant_drag]
        ])
        B_inv = np.linalg.inv(B)
        omega_motor_square = B_inv @ np.concatenate([np.array([thrust]), tau])
        omega_motor = np.sqrt(np.clip(omega_motor_square, 0, None))
        self.omega_motors = omega_motor
        return omega_motor
    
    def save_trajectory(self, filename):
        fields = ['t', 'p', 'p_d', 'v', 'v_d', 'q', 'R', 'w', 'T_sp', 'q_sp', 'hover_throttle', 'fa', 'pwm']
        # Create a DataFrame from self.traj (which is assumed to be a list of rows in the same order as fields).
        df = pd.DataFrame(self.traj, columns=fields)
        # ,t,p,p_d,v,v_d,q,R,w,T_sp,q_sp,hover_throttle,fa,pwm
        df.to_csv(filename, index=True, index_label="", quoting=csv.QUOTE_MINIMAL)
        print("Trajectory saved to", filename)

PLAYBACK_SPEED = 1
CONTROL_FREQUENCY = 200 # Hz for attitude control loop
dt = 1.0 / CONTROL_FREQUENCY
time = [0.0]

def get_pos_full_quadcopter(quad):
    """ position returns a 3 x 6 matrix 
        where row is [x, y, z] column is m1 m2 m3 m4 origin h
    """
    origin = quad.state[IDX_POS_X:IDX_POS_Z+1]
    quat = quad.state[IDX_QUAT_W:IDX_QUAT_Z+1]
    rot = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True).as_matrix()
    wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
    quadBodyFrame = quad.body_frame.T
    quadWorldFrame = wHb.dot(quadBodyFrame)
    pos_full_quad = quadWorldFrame[0:3]
    return pos_full_quad

def control_propellers(quad, model):
    # t = quad.time
    # T = 10.0
    # r = 2*np.pi * t / T
    # prop_thrusts = quad.control(p_d_I = np.array([np.cos(r/2), np.sin(r), 0.0]))
    # Note: for Hover mode, just replace the desired trajectory with [1, 0, 1]

    prop_thrusts = quad.control(p_d_I = np.array([0.0, 0.0, 0.0]), v_d_I = np.array([0.0, 0.0, 0.0]), model=model)
    quad.update(prop_thrusts, dt)

def sim_without_animation(model):
    quadcopter = Robot()
    for _ in range(2000):
        control_propellers(quadcopter,model)
    quadcopter.save_trajectory("data/custom_constant_baseline_6wind.csv")

def sim_with_animation(model):
    quadcopter = Robot()
    def control_loop(i):
        for _ in range(PLAYBACK_SPEED):
            control_propellers(quadcopter, model)
        return get_pos_full_quadcopter(quadcopter)

    plotter = QuadPlotter()
    plotter.plot_animation(control_loop)

def main():
    #Load in model 
    dim_a = 3
    features = ['v', 'q', 'pwm']
    dataset = 'sim' 
    #Load in model 
    modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}"
    stopping_epoch = 50
    model = mlmodel.load_model(modelname = modelname + '-epoch-' + str(stopping_epoch))
    #print("Loaded model: ",model)
    sim_with_animation(model)    
    #sim_without_animation(model)

if __name__ == "__main__":
    main()