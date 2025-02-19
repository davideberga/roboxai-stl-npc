from alg.RoverSTL import RoverSTL
import numpy as np
import config

# Call the main function
if __name__ == "__main__":
    # Default parameters
    args = config.parse_args()
    
    algo = RoverSTL(None, args)
    algo.generate_dataset(args)
    
    # l = np.load('dataset/dynamics-states-1739821779.npz', allow_pickle=True)
    # print(l["world_objects"])
    # print(l["states"].shape)
