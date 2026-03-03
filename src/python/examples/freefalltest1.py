import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
# Number of points (stars) to plot - Increased significantly for arm definition
NUM_POINTS = 10000 
# Max radius of the galaxy disk
MAX_RADIUS = 5.0
# Speed of rotation (degrees per frame)
ROTATION_SPEED = 0.5 # Slowed down for better viewing
# Number of frames in the animation
NUM_FRAMES = 720 

# --- Physical Parameters for Spiral Arms ---
NUM_ARMS = 2 # Try changing this value (e.g., to 2, 3, or 5)
# Pitch angle of the logarithmic spiral, common value for spirals like the Milky Way
PITCH_ANGLE_DEG = 12 
PITCH_ANGLE_RAD = np.deg2rad(PITCH_ANGLE_DEG)
# Tangent of the pitch angle (used in the logarithmic spiral formula)
TAN_PITCH = np.tan(PITCH_ANGLE_RAD)

# Angular offsets for each arm (dynamically calculated and randomized for asymmetry)
# 1. Calculate base symmetrical separation (e.g., 0, 90, 180, 270 degrees for 4 arms)
base_separation = np.linspace(0, 2 * np.pi, NUM_ARMS, endpoint=False)

# 2. Add controlled random perturbation to each arm's starting angle (asymmetry)
# A small shift (+/- 0.2 radians) breaks perfect symmetry while keeping structure
np.random.seed(42) # Seed for consistent visualization of the asymmetry
asymmetry_perturbation = (np.random.rand(NUM_ARMS) - 0.5) * 0.4 

# Combine to form the final offsets array
ARM_OFFSETS = base_separation + asymmetry_perturbation

# Scatter factor: how tightly stars cluster around the arm's ideal position (lower is tighter)
ARM_SCATTER = 0.05 


# --- Data Generation for the Galaxy ---
def generate_galaxy_data(num_points, max_radius):
    """
    Generates coordinates for a galaxy using a physically motivated logarithmic 
    spiral model with density concentration.
    """
    
    # 1. Central Bulge (Inner, thicker, more uniform part)
    # Keeping the core very small and flat
    num_bulge_points = num_points // 50
    
    # Generate Bulge data (same as before)
    bulge_r = np.random.normal(0, 0.1, num_bulge_points) * max_radius * 0.5
    bulge_theta = np.random.rand(num_bulge_points) * 2 * np.pi
    bulge_z = np.random.normal(0, 0.1, num_bulge_points) * max_radius * 0.1
    bulge_x = bulge_r * np.cos(bulge_theta)
    bulge_y = bulge_r * np.sin(bulge_theta)
    
    # 2. Disk and Spiral Arms 
    num_disk_points = num_points - num_bulge_points
    
    # Radius distribution for the disk (emphasizes outer regions)
    r = np.random.power(1.8, num_disk_points) * max_radius
    r[r < 0.1] = 0.1 # Prevent log(r) issues near zero
    
    # Assign each star to one of the arms
    arm_assignment = np.random.randint(0, NUM_ARMS, num_disk_points)
    
    # Calculate the base angle shift due to the logarithmic spiral
    # Logarithmic Spiral Formula: phi = (1/tan(alpha)) * log(r)
    log_spiral_angle = np.log(r / 0.1) / TAN_PITCH
    
    # Calculate the final angle (theta) for each star
    theta = np.zeros(num_disk_points)
    
    for i in range(NUM_ARMS):
        # Find points belonging to the current arm
        arm_indices = arm_assignment == i
        
        # Base position of the arm
        arm_base_pos = log_spiral_angle[arm_indices] + ARM_OFFSETS[i]
        
        # Scatter: Add small Gaussian noise to concentrate stars along the arm's path
        scatter = np.random.normal(0, ARM_SCATTER, np.sum(arm_indices))
        
        # Final angular position: Arm Position + Scatter Noise
        theta[arm_indices] = arm_base_pos + scatter
        
    # X and Y coordinates calculation
    disk_x = r * np.cos(theta)
    disk_y = r * np.sin(theta)
    
    # Z-height (Very thin disk, proportional to radius)
    z = np.random.normal(0, 0.005, num_disk_points) * r # Reduced Z-noise
    
    # Combine data
    R = np.concatenate([bulge_r, r])
    X = np.concatenate([bulge_x, disk_x])
    Y = np.concatenate([bulge_y, disk_y])
    Z = np.concatenate([bulge_z, z])

    return X, Y, Z, R

# --- Setup the Plot ---
# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Logarithmic Spiral Holographic Galaxy (Physically Motivated)", color='cyan', fontsize=16)

# Set the background to dark/black for a "holographic" feel
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Hide the axis labels and ticks for a cleaner view
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_axis_off()

# Set initial view to the best angle to see the arms (low elevation)
# This low elevation (10 degrees) is critical for seeing the spiral structure
ax.view_init(elev=10, azim=0) 

# Set equal aspect ratio to ensure the galaxy shape is correct
max_range = MAX_RADIUS * 1.1
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range * 0.3, max_range * 0.3]) 

# Generate galaxy coordinates
X, Y, Z, R = generate_galaxy_data(NUM_POINTS, MAX_RADIUS)

# Plot the galaxy as scattered points.
galaxy_model = ax.scatter(
    X, Y, Z, 
    c=R,            # Color based on radial distance
    cmap='inferno', # Brighter colormap to emphasize density (Yellow-Orange-Red)
    s=1,            # Small point size for a sharp, dense look
    alpha=0.8, 
    marker='.'
)

# Keep a small central point for focus.
ax.scatter([0], [0], [0], color='white', s=30, alpha=1.0)


# --- Animation Function ---
def update_galaxy(frame):
    """Updates the view angle to simulate rotation."""
    new_azim = frame * ROTATION_SPEED
    
    # Maintain the low elevation (10 degrees) while rotating the azimuth
    ax.view_init(elev=45, azim=new_azim)
    
    return galaxy_model,

# --- Run Animation ---
ani = FuncAnimation(
    fig, 
    update_galaxy, 
    frames=NUM_FRAMES, 
    interval=10, # Faster interval for smoother rotation
    blit=False, 
    repeat=True
)

# Display the plot
print("Generating 3D galaxy animation with Logarithmic Spiral structure.")
print("The arms should now be much clearer due to star concentration.")
plt.show()