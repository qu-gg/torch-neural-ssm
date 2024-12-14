import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_ball_trajectory(images):
    # Define number of timesteps and image dimensions
    timesteps, dim1, dim2 = images.shape

    # Use 'turbo' colormap for a bright and high-contrast gradient
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=2, vmax=timesteps - 1)
    sm = ScalarMappable(cmap=cmap, norm=norm)

    # Create a base figure for the trajectory plot
    plt.figure(figsize=(6, 6))
    plt.axis("off")

    # Plot each timestep's image on the plot with a color gradient
    for t in range(timesteps):
        # Get color for the current timestep
        color = cmap(norm(t))[:3]  # RGB color for the current timestep

        # Create an RGB mask for the current frame
        mask = images[t] > 0  # Assuming ball is represented by non-zero values
        colored_image = np.zeros((dim1, dim2, 3))  # Create an RGB image
        for c in range(3):  # Assign the color to masked areas only
            colored_image[:, :, c] = mask * color[c]

        # Plot the colored mask with some transparency
        plt.imshow(1 - colored_image, alpha=0.3)

    # Add color bar to indicate time progression
    plt.colorbar(sm, label="Time Steps")
    plt.title("Ball Trajectory Over Time", color='white')
    plt.gca().set_facecolor('black')  # Set the background to black for contrast
    plt.show()

images = np.load("bouncingball_10000samples_40steps/train.npz", allow_pickle=True)["image"]
plot_ball_trajectory(images[0])
plot_ball_trajectory(images[1])
plot_ball_trajectory(images[2])