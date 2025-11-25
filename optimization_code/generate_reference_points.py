import numpy as np
from itertools import combinations_with_replacement

def generate_reference_points(n_obj, H):
    """
    Generates reference points for n-objective optimization problems using the method
    proposed by Das and Dennis. This function is generalized for any number of objectives.

    Args:
        n_obj (int): The number of objectives.
        H (int): The number of divisions along each objective axis.

    Returns:
        np.ndarray: An array of reference points, shape (n_points, n_obj).
    """
    if n_obj <= 0 or H <= 0:
        return np.array([])
        
    # The number of reference points is C(H + n_obj - 1, n_obj - 1)
    # This is equivalent to finding the number of non-negative integer solutions to
    # x_1 + x_2 + ... + x_{n_obj} = H
    
    # We can use combinations with replacement to generate the numerators
    # of the reference points.
    # Imagine H items to be distributed into n_obj bins.
    # This is equivalent to arranging H identical items and n_obj-1 dividers.
    
    # The combinations_with_replacement function from itertools is perfect for this.
    # It generates tuples of length n_obj that sum up to H.
    
    # Generate combinations of indices
    combs = combinations_with_replacement(range(H + n_obj - 1), n_obj - 1)
    
    points = []
    for c in combs:
        point = np.zeros(n_obj)
        
        # Convert combination indices to a point on the simplex
        # This is a standard way to map combinations to simplex points
        # See: https://stackoverflow.com/questions/35493341/how-to-generate-a-set-of-vectors-in-a-unit-simplex
        
        # Create a temporary array with 0, H+n_obj-1, and the combination indices
        temp = np.concatenate(([0], np.array(c) + 1, [H + n_obj - 1]))
        
        # The differences between consecutive elements give the numerators
        diffs = np.diff(temp) - 1
        
        # Normalize to get the final reference point
        point = diffs / H
        points.append(point)
        
    # A simpler, more direct method using a recursive approach or direct generation
    # Let's use a more intuitive recursive generator
    
    def get_points_recursive(n_obj_left, H_left, current_point):
        if n_obj_left == 1:
            yield current_point + [H_left]
        else:
            for i in range(H_left + 1):
                yield from get_points_recursive(n_obj_left - 1, H_left - i, current_point + [i])

    points = np.array(list(get_points_recursive(n_obj, H, []))) / H
    
    return points


if __name__ == '__main__':
    # Example usage:
    
    # --- 2 Objectives ---
    H_2d = 13
    n_obj_2d = 2
    ref_points_2d = generate_reference_points(n_obj_2d, H_2d)
    print(f"Generated {ref_points_2d.shape[0]} reference points for {n_obj_2d} objectives with H={H_2d}.")
    # Expected: H+1 = 14 points
    
    # --- 3 Objectives ---
    H_3d = 13
    n_obj_3d = 3
    ref_points_3d = generate_reference_points(n_obj_3d, H_3d)
    print(f"Generated {ref_points_3d.shape[0]} reference points for {n_obj_3d} objectives with H={H_3d}.")
    # Expected: C(13+3-1, 3-1) = C(15, 2) = 105 points

    # --- 4 Objectives ---
    H_4d = 7
    n_obj_4d = 4
    ref_points_4d = generate_reference_points(n_obj_4d, H_4d)
    print(f"Generated {ref_points_4d.shape[0]} reference points for {n_obj_4d} objectives with H={H_4d}.")
    # Expected: C(7+4-1, 4-1) = C(10, 3) = 120 points

    # Visualization for 3D case (optional, requires matplotlib)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ref_points_3d[:, 0], ref_points_3d[:, 1], ref_points_3d[:, 2])
    ax.set_title(f'Reference Points for 3 Objectives (H={H_3d})')
    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.set_zlabel('Objective 3')
    ax.view_init(30, 30)
    plt.show()
