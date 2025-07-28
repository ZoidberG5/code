import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

class RadarDisplay:
    def __init__(self, objects):
        """
        Initialize the Radar Display with a list of objects.
        
        :param objects: List of matched objects, each with 'distance' and 'angle' attributes.
        """
        # Set up the Tkinter window
        self.root = tk.Tk()
        self.root.title("Radar Display")

        # Create a matplotlib figure for the radar
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize radar properties
        self.ax.set_theta_zero_location("N")  # Set 0° at the top (north)
        self.ax.set_ylim(0, 15)  # Max distance (customize based on actual distance range)
        self.ax.set_thetamin(-30)  # Display from -45° to 45°
        self.ax.set_thetamax(30)
        self.ax.set_xticks(np.linspace(-np.pi / 4, np.pi / 4, 3))  # Only -45°, 0°, 45°
        self.ax.set_xticklabels(['-30°', '0°', '30°'])

        self.objects = objects
        self.update_display(objects)  # Initial display

    def update_display(self, objects):
        """
        Update the radar display with new matched objects.

        :param objects: List of matched objects, each with 'distance' and 'angle' attributes.
        """
        # Clear previous points
        self.ax.clear()

        # Reapply radar grid settings
        self.ax.set_theta_zero_location("N")
        self.ax.set_ylim(0, 15)
        self.ax.set_thetamin(-30)
        self.ax.set_thetamax(30)
        self.ax.set_xticks(np.linspace(-np.pi / 4, np.pi / 4, 3))
        self.ax.set_xticklabels(['-30°', '0°', '30°'])

        # Plot each object
        for obj in objects:
            angle_rad = np.radians(obj['angle'])  # Convert angle to radians
            angle_rad = -angle_rad
            self.ax.plot(angle_rad, obj['distance'], 'bo')  # Plot object as a blue dot
            self.ax.text(angle_rad, obj['distance'] + 0.5, {obj['name'], obj['distance']}, ha='center', color='blue', fontsize=8)

        # Redraw the canvas
        self.canvas.draw()
    
    def run(self):
        self.root.mainloop()
