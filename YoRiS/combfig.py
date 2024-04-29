import matplotlib.pyplot as plt
from matplotlib.image import imread

# File paths for the 11 figures
file_paths = [
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR1.png',
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR2.png',
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR3.png',
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR4.png',
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR5.png',
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR6.png',
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR7.png',
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR8.png',
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR9.png',
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR10.png',
    'C:\\Users\\aust_\\OneDrive\\Desktop\\figures\\FR11.png',
]

# Create a 3x4 subplot grid (adjust rows and columns as needed)
rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(12, 9), dpi=300)  # Adjust DPI here

# Iterate over file paths and plot each image in the subplot grid
for i, file_path in enumerate(file_paths):
    row = i // cols
    col = i % cols
    img = imread(file_path)
    axes[row, col].imshow(img)
    axes[row, col].axis('off')

# Adjust layout for better spacing
plt.subplots_adjust(wspace=0, hspace=0)  # Adjust spacing between subplots
plt.tight_layout()

# Adjust font size
font_size = 14
for ax in axes.flat:
    ax.label_outer()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)

# Show the combined subplot
plt.show()
