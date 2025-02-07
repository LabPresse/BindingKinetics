
# Import libraries
import sys
import numba as nb
from main_states import main_states
from main_rates import main_rates


# Main script
if __name__ == "__main__":
    ID = 0
    if len(sys.argv) > 1:
        ID = int(sys.argv[1])

    # Uncomment the following line corresponding to the function you want to run
    # main_states(ID)
    main_rates(ID)
