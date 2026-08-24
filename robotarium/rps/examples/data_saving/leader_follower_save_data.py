import numpy as np
import time
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics, create_si_to_uni_mapping
from rps.utilities.graph import completeGL, topological_neighbors
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 4  # 1 leader + 3 followers
iterations = 5000 

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
# Using the updated NumberOfRobots and InitialConditions kwargs
initial_conditions = np.array([
    [0, 0.5, 0.3, -0.1],
    [0.5, 0.5, 0.2, 0],
    [0, 0, 0, 0]
])

r = robotarium.Robotarium(
    number_of_robots=N, 
    show_figure=True, 
    initial_conditions=initial_conditions, 
    sim_in_real_time=False
)

# =========================================================
# GRAPH LAPLACIAN AND FORMATION SETUP
# =========================================================
# Followers form a complete graph; Robot 1 (Python index 1) connects to Leader (index 0)
followers = -completeGL(N-1)
L = np.zeros((N, N))
L[1:N, 1:N] = followers
L[1, 1] += 1
L[1, 0] = -1

# Identify edges for data recording (matching MATLAB rows/cols logic)
[rows, cols] = np.where(L == 1)

formation_control_gain = 10
desired_distance = 0.2 # Matches MATLAB version

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
# Updated to match MATLAB gains (0.8)
si_to_uni_dyn = create_si_to_uni_dynamics(linear_velocity_gain=0.8)
uni_barrier_cert = create_uni_barrier_certificate_with_boundary()
leader_controller = create_si_position_controller(
    x_velocity_gain=0.8, 
    y_velocity_gain=0.8, 
    velocity_magnitude_limit=0.1
)

# =========================================================
# WAYPOINT INITIALIZATION
# =========================================================
waypoints = np.array([[-1, -1, 1, 1], [0.8, -0.8, -0.8, 0.8]])
close_enough = 0.03
state = 0 # Leader starts targeting waypoint 1 (index 0)

# =========================================================
# DATA SAVING SETUP
# =========================================================
# Row 1: Leader to Robot 2 | Rows 2-4: Follower pairs | Row 5: Time
robot_distance_log = np.zeros((5, iterations))
goal_distance_log = [] 
start_time = time.time()

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for t in range(iterations):
    x = r.get_poses()
    dxi = np.zeros((2, N))

    # 1. Formation control for followers (indices 1 to N-1)
    for i in range(1, N):
        neighbors = topological_neighbors(L, i)
        for j in neighbors:
            # Potential field law: gain * (||dist||^2 - desired^2) * relative_pos
            dist_sq = np.linalg.norm(x[:2, j] - x[:2, i])**2
            dxi[:, i] += formation_control_gain * (dist_sq - desired_distance**2) * (x[:2, j] - x[:2, i])

    # 2. Leader waypoint tracking
    waypoint = waypoints[:, state].reshape(2, 1)
    dxi[:, 0] = leader_controller(x[:2, 0].reshape(2, 1), waypoint).flatten()

    if np.linalg.norm(x[:2, 0].reshape(2, 1) - waypoint) < close_enough:
        # Record leader-to-goal distance when "close enough"
        goal_dist = np.linalg.norm(x[:2, 0].reshape(2, 1) - waypoint)
        goal_distance_log.append([goal_dist, time.time() - start_time])
        state = (state + 1) % waypoints.shape[1]

    # 3. Velocity thresholding for followers (matching 3/4 max velocity logic)
    norms = np.linalg.norm(dxi, axis=0)
    threshold = 0.75 * r.MAX_LINEAR_VELOCITY
    to_thresh = (norms > threshold)
    to_thresh[0] = False # Leader is excluded from thresholding
    
    if np.any(to_thresh):
        dxi[:, to_thresh] *= threshold / norms[to_thresh]

    # 4. Conversion and Safety
    dxu = si_to_uni_dyn(dxi, x)
    dxu = uni_barrier_cert(dxu, x)

    r.set_velocities(np.arange(N), dxu)

    # 5. Record distance data
    # Leader (0) to Robot 2 (1)
    robot_distance_log[0, t] = np.linalg.norm(x[:2, 0] - x[:2, 1])
    # Follower-Follower pairs based on graph edges
    num_follower_edges = int(len(rows)/2)
    for b in range(num_follower_edges):
        robot_distance_log[b+1, t] = np.linalg.norm(x[:2, rows[b]] - x[:2, cols[b]])
    
    robot_distance_log[4, t] = time.time() - start_time

    r.step()

# =========================================================
# SAVE DATA
# =========================================================
# Saving as .npy (binary) and .txt (CSV) to match your Python requirement
goal_data_final = np.array(goal_distance_log)

np.save('inter_robot_distance_data.npy', robot_distance_log)
np.save('goal_distance_data.npy', goal_data_final)

np.savetxt('inter_robot_distance_data.txt', robot_distance_log.T, delimiter=',')
np.savetxt('goal_distance_data.txt', goal_data_final, delimiter=',')

r.debug()