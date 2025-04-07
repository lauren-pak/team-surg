import numpy as np #numerical operations
import pickle #read and write python objects in binary
import os #file handling
import torch #deep learning library
import io # in-memory file-like ojvects
from util import read_pickle, apply_transform

def compute_gaze_vector(joints):
    """
    Computes the gaze direction vector from a single person's joint coordinates.

    Returns: (origin, gaze_vector): A tuple where `origin` is the midpoint between the eyes,
                                    and `gaze_vector` is the normalized gaze direction.
    """
    L = joints[57] #left eye position
    R = joints[56] #right eye position
    N = joints[12] #nose position

    mu = (L + R) / 2  # Midpoint between eyes
    initial_gaze = mu - N
        
    #Plane - a flat surface that extends infinetly in all directions, like a graph.
    # Define a plane using the vector between the eyes and the initial gaze
    eye_vector = L - R

    #cross product (third perpendicular vector) to both vectors
    plane_normal = np.cross(eye_vector, initial_gaze) 

    #normalising the vector - scaling it so that its length (magnitude) becomes 1
    #easier to work with with a unit length
    plane_normal = plane_normal / np.linalg.norm(plane_normal)


    # Projection of the initial gaze onto the plane

    #calculates how much of the component initial_gaze is along the plane normal then removes the part that is pointing out of the plane
    corrected_gaze = initial_gaze - np.dot(initial_gaze, plane_normal) * plane_normal
    corrected_gaze = corrected_gaze / np.linalg.norm(corrected_gaze) #normalising
    return mu, corrected_gaze

def count_focused_attention_events(data_dict, margin_of_error, time_frame, start_frame_dict, total_num_frames, step_size=3):
    """
        Counts how many times a person's gaze remains stable over a given time
    """
    events = {}
    window_size = time_frame 
    event_record = [0 for _ in range(total_num_frames)]
    
    for person_id, person_data in data_dict.items():
        start_frame = start_frame_dict[person_id] #retrieve joint data
        person_data = np.array(person_data) #turn data into array
        T = person_data.shape[0] #number of frames for this person

        # Compute gaze vectors (direction component) for each time frame.
        gaze_directions = np.zeros((T, 3))
        for t in range(T):
            _, gaze_vector = compute_gaze_vector(person_data[t])
            gaze_directions[t] = gaze_vector
        event_count = 0
        
        # sliding window
        for t in range(0, T - window_size + 1, step_size):
            window = gaze_directions[t:t+window_size]
            ref_vector = window[0] #first gaze vector

            # Compute the dot product between the first vector and each vector in the window. (clip just keeps bounds consistent for doing inverse cosine)
            dots = np.clip(np.sum(window * ref_vector, axis=1), -1.0, 1.0)
            #when the gaze remains constant/the same as the ref

            # Calculate angle differences (in radians)
            angle_deviations_rad = np.arccos(dots)
            angle_deviations_deg = np.degrees(angle_deviations_rad) 
            # If all angles in this window are within the allowed margin, count an event
            # means person is not disracted

            if np.all(angle_deviations_deg <= margin_of_error):
                # print("event: time " + str(t))
                event_count += 1
                for frame_idx in range(start_frame, start_frame + window_size): #Update event record 
                    #what frames are the person staring in the right direction
                    event_record[frame_idx] += 1
                    
        events[person_id] = event_count
        
    return events, event_record 

def process_files(folder_path, debug=False):
    #returns dictionary where keys are IDs and values are in the shape [T, J, 3]
    paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')])
    #stores total numbers of frames
    if debug:
        paths = paths[4930:]
    total_num_frames = len(paths) 
    data = {} #stores joint positions
    start_frame_dict = {} #stores each person's starting frames
    for frame_idx, path in enumerate(paths):
        f = read_pickle(path)
        joints, vertices = apply_transform(f, True)

        for i, tracker_id in enumerate(f['trackers']): #extracts each persons joints
            person_joints = joints[i]
            if tracker_id in data:
                data[tracker_id].append(person_joints)
            else:
                data[tracker_id] = [person_joints]
                start_frame_dict[tracker_id] = frame_idx   
    return data, start_frame_dict, total_num_frames

if __name__ == "__main__":
    print("starting")


dataset = read_pickle("empirical/joint_out/frame_000450.pkl")
data, start_frames, total_frames = process_files(dataset)
