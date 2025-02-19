from alg.RoverSTL import RoverSTL
import config


# Call the main function
if __name__ == "__main__":
    # Default parameters
    args = config.parse_args()

    # seed = None implies random seed
    editor_build = True
    env_type = "training"

    print("STL Rover training with Davide and Martina!\n")
    
    algo = RoverSTL(None, args)
    algo.training_loop(args)

