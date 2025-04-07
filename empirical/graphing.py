import os
import torch
import pickle
import util
import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
sys.path.append(r'/Users/laurenpak/Desktop/Coding/GitHub/team-surg/lightning')
from joints import JOINT_NAMES

# --- Set up paths and working directory ---
base_path = r"/Users/laurenpak/Desktop/Coding/GitHub/team-surg/empirical/joint_out"
directory_path = '/Users/laurenpak/Desktop/Coding/GitHub/team-surg/temp'  # Use forward slashes or double backslashes
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory created: {directory_path}")
else:
    print(f"Directory already exists: {directory_path}")
os.chdir(directory_path)
print("Changed to new Working Directory:", os.getcwd())

# --- Load main data file ---
file_path = r'/Users/laurenpak/Desktop/Coding/GitHub/team-surg/empirical/joint_out/frame_000020.pkl'
try:
    data = util.read_pickle(file_path)
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print("Error:", e)

print("Current Working Directory:", os.getcwd())
print(data)
print(os.listdir())
print(type(data))
print(data.keys() if isinstance(data, dict) else "Not a dictionary")




#LOADING THE DATA

joints = data['joints3d']
vertices = data['vertices']
print(len(joints))
print(joints.shape)
print(vertices.shape)

# --- Rotation matrix function ---
def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux*uy*(1 - cos_theta) - uz*sin_theta, ux*uz*(1 - cos_theta) + uy*sin_theta],
        [uy*ux*(1 - cos_theta) + uz*sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy*uz*(1 - cos_theta) - ux*sin_theta],
        [uz*ux*(1 - cos_theta) - uy*sin_theta, uz*uy*(1 - cos_theta) + ux*sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])
    return R

person_data = []

'''centering'''
# rotation_matrix180 = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])
# xz_matrix = rotation_matrix(np.array([0,1,0]), np.radians(10))
# yz_matrix = rotation_matrix(np.array([1,0,0]), np.radians(-75))
# matrix_final = np.dot(rotation_matrix180, xz_matrix)
# matrix_final = np.dot(matrix_final, yz_matrix)

# for i in range(min(len(joints), 5)):
#     j3d = joints[i].cpu().numpy() 
#     v3d = vertices[i].cpu().numpy()
#     j3d = j3d @ matrix_final
#     v3d = v3d @ matrix_final
#     person_data.append((j3d, v3d))


'''without transformation function'''
for i in range(min(len(joints), 5)):
    j3d = joints[i].cpu().numpy()
    v3d = vertices[i].cpu().numpy()
    person_data.append((j3d, v3d))



# --- Scatter plots ---
#plot the entire body
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['b', 'r', 'g', 'y', 'm']
for i, (j3d, v3d) in enumerate(person_data):
    ax.scatter(v3d[:,0], v3d[:,1], v3d[:,2], c=colors[i], marker='o', s=50, label=f'3D Vertices Person {i+1}', zorder=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d([-0.5, 0.5])
ax.view_init(elev=10, azim=90)
ax.legend()
plt.show()



#plotting all joints of a person
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
for i, (j3d, _) in enumerate(person_data):
    ax2.scatter(j3d[:,0], j3d[:,1], j3d[:,2], marker='x', c=colors[i], label=f'3D Joints Person {i+1}')
ax2.view_init(elev=10, azim=0)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()
plt.show()

print(person_data[0][1].mean(axis=0))
print(person_data[0][1].std(axis=0))
print(person_data[0][0].mean(axis=0))
print(person_data[0][0].std(axis=0))



# --- Function to display frames for one person ---
def displayframesoneperson(frames, num_frames, count_by, person_index):
    num_frames = min(num_frames, len(frames))
    frame_indices = [i * count_by for i in range(num_frames) if i * count_by < len(frames)]
    rows = math.ceil(num_frames / 2)
    cols = 2
    rotation_matrix180 = np.array([[-1,0,0],[0,-1,0], [0,0,-1]])
    xz_matrix = rotation_matrix(np.array([0,1,0]), np.radians(10))
    yz_matrix = rotation_matrix(np.array([1,0,0]), np.radians(-10))
    matrix_final = rotation_matrix180 @ xz_matrix @ yz_matrix
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10), subplot_kw={'projection': '3d'})
    axs = axs.flatten()
    for j, i in enumerate(frame_indices):
        data = frames[i]
        joints = data['joints3d']
        vertices = data['vertices']


        j3d = joints[person_index].cpu().numpy()
        v3d = vertices[person_index].cpu().numpy()
        '''TRANSFORMATION FUNCTION'''
        # j3d = joints[person_index].cpu().numpy() - joints[person_index].cpu().numpy().mean(axis=0)
        # v3d = vertices[person_index].cpu().numpy() - vertices[person_index].cpu().numpy().mean(axis=0)
        # j3d = j3d @ matrix_final
        # v3d = v3d @ matrix_final

        axs[j].scatter(v3d[:, 0], v3d[:, 1], v3d[:, 2], c='b', marker='o', s=50, label='3D Vertices', zorder=1, depthshade=False)
        axs[j].scatter(j3d[:, 0], j3d[:, 1], j3d[:, 2], c='magenta', marker='x', s=50, label='3D Joints', zorder=2, depthshade=False)
        axs[j].set_xlabel('X')
        axs[j].set_ylabel('Y')
        axs[j].set_zlabel('Z')
        axs[j].legend()
        axs[j].set_title(f'Frame {i}, Person {person_index}')
    for k in range(len(frame_indices), len(axs)):
        fig.delaxes(axs[k])
    plt.tight_layout()
    plt.show()

# --- Function to plot individual joint ---
def plot_individual_joint(frames, joint_name, person_index=0):
    if not frames:
        print("Error: No frames loaded.")
        return

    try:
        joint_index = JOINT_NAMES.index(joint_name)
    except ValueError:
        print(f"Error: Joint '{joint_name}' not found in JOINT_NAMES.")
        return

    joints = frames[0]['joints3d']  # Use first frame for simplicity
    num_persons = joints.shape[0]
    print(f"Total persons detected: {num_persons}")

    if num_persons == 0:
        print("Error: No persons found in the data.")
        return

    if person_index >= num_persons:
        print(f"Warning: person_index {person_index} is too high. Using last available index {num_persons - 1}.")
        person_index = num_persons - 1  # Use last available person

    joint_pos = joints[person_index][joint_index].cpu().numpy()

    # Plot the joint position
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joint_pos[0], joint_pos[1], joint_pos[2], color='red', s=100, label=joint_name)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Position of {joint_name} (Person {person_index})")
    ax.legend()

    plt.show()

# --- Load multiple frames and display ---
frame_paths = [r'/Users/laurenpak/Desktop/Coding/GitHub/team-surg/empirical/joint_out/frame_000120.pkl', r'/Users/laurenpak/Desktop/Coding/GitHub/team-surg/empirical/joint_out/frame_000590.pkl', r'/Users/laurenpak/Desktop/Coding/GitHub/team-surg/empirical/joint_out/frame_000740.pkl']
frames = [util.read_pickle(fp) for fp in frame_paths]


displayframesoneperson(frames, 3, 1, 0)
plot_individual_joint(frames, 'left_shoulder', 2)