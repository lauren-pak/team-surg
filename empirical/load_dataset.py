from util import apply_transform, apply_padding, read_pickle


dataset = read_pickle("empirical/joint_out/frame_000450.pkl") #chose a random frame

joints, verticies = apply_transform(dataset)
padded_data = apply_padding(joints)


print(padded_data)