import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

origin = (45.4227, 55.4867-60)

# Load the image
image_path = 'calibration.png'  # Replace with your image file path
image = mpimg.imread(image_path)

data = pd.read_csv('cylinder0001.csv', delimiter=';')
x = data['x [mm]'].to_numpy() + origin[0]
y = data['y [mm]'].to_numpy() + origin[1]+60

# Rescale the image
scale_factor = 8 / (1295.01 - 1101.01)
rescaled_image = plt.imshow(image, extent=[0, image.shape[1] * scale_factor, 0, image.shape[0] * scale_factor])

# Plot the rescaled image
plt.imshow(rescaled_image.get_array(), extent=rescaled_image.get_extent())
plt.axis('off')  # Hide the axis


plt.plot(*origin, 'bo')  # Plot the origin

circle = plt.Circle(origin, 25.4+5, color='r', fill=False)
plt.gca().add_patch(circle)

circle = plt.Circle(origin, 25.4, color='b', fill=False)
plt.gca().add_patch(circle)

plt.plot(x, y, 'o')

plt.show()