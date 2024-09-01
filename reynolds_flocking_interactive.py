import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

CONTAINER_SIZE = 30

class Boid:
    def __init__(self, x, y, vx, vy):
        self.position = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.acceleration = np.zeros(2)

    def update(self, dt):
        self.velocity += self.acceleration * dt
        self.velocity = np.clip(self.velocity, -max_speed.val, max_speed.val)
        self.position += self.velocity * dt
        self.acceleration = np.zeros(2)

    def avoid_walls(self, margin=2):
        desired = None
        if self.position[0] < margin:
            desired = np.array([max_speed.val, self.velocity[1]])
        elif self.position[0] > CONTAINER_SIZE - margin:
            desired = np.array([-max_speed.val, self.velocity[1]])
        
        if self.position[1] < margin:
            desired = np.array([self.velocity[0], max_speed.val])
        elif self.position[1] > CONTAINER_SIZE - margin:
            desired = np.array([self.velocity[0], -max_speed.val])
        
        if desired is not None:
            steer = desired - self.velocity
            steer = steer / np.linalg.norm(steer) if np.linalg.norm(steer) > 0 else steer
            return steer
        return np.zeros(2)

class Flock:
    def __init__(self, num_boids):
        self.boids = [Boid(np.random.rand() * CONTAINER_SIZE, np.random.rand() * CONTAINER_SIZE, 
                           np.random.randn() * 2, np.random.randn() * 2) for _ in range(num_boids)]
        self.num_boids = num_boids

    def apply_behavior(self, dt):
        for boid in self.boids:
            separation = self.separation(boid) * separation_weight.val
            alignment = self.alignment(boid) * alignment_weight.val
            cohesion = self.cohesion(boid) * cohesion_weight.val
            wall_avoidance = boid.avoid_walls() * wall_avoidance_weight.val
            
            boid.acceleration = separation + alignment + cohesion + wall_avoidance
            boid.update(dt)
            
            # Ensure boids stay within boundaries
            boid.position = np.clip(boid.position, 0, CONTAINER_SIZE)

    def separation(self, boid):
        steering = np.zeros(2)
        for other in self.boids:
            if other != boid:
                diff = boid.position - other.position
                dist = np.linalg.norm(diff)
                if 0 < dist < perception_radius.val:
                    steering += diff / (dist * dist)
        return steering

    def alignment(self, boid):
        steering = np.zeros(2)
        total = 0
        for other in self.boids:
            if other != boid:
                if np.linalg.norm(boid.position - other.position) < perception_radius.val:
                    steering += other.velocity
                    total += 1
        if total > 0:
            steering /= total
            steering -= boid.velocity
        return steering

    def cohesion(self, boid):
        steering = np.zeros(2)
        total = 0
        for other in self.boids:
            if other != boid:
                if np.linalg.norm(boid.position - other.position) < perception_radius.val:
                    steering += other.position
                    total += 1
        if total > 0:
            steering /= total
            return (steering - boid.position) / 100
        return steering

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(bottom=0.3)
ax.set_xlim(0, CONTAINER_SIZE)
ax.set_ylim(0, CONTAINER_SIZE)

# Create sliders
slider_color = 'lightgoldenrodyellow'
slider_ax_container = plt.axes([0.2, 0.20, 0.6, 0.03], facecolor=slider_color)
slider_ax_speed = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor=slider_color)
slider_ax_perception = plt.axes([0.2, 0.10, 0.6, 0.03], facecolor=slider_color)
slider_ax_separation = plt.axes([0.2, 0.05, 0.15, 0.03], facecolor=slider_color)
slider_ax_alignment = plt.axes([0.35, 0.05, 0.15, 0.03], facecolor=slider_color)
slider_ax_cohesion = plt.axes([0.5, 0.05, 0.15, 0.03], facecolor=slider_color)
slider_ax_wall = plt.axes([0.65, 0.05, 0.15, 0.03], facecolor=slider_color)

max_speed = Slider(slider_ax_speed, 'Max Speed', 0.1, 5, valinit=2)
perception_radius = Slider(slider_ax_perception, 'Perception Radius', 0.1, 10, valinit=2)
separation_weight = Slider(slider_ax_separation, 'Separation', 0, 5, valinit=1.5)
alignment_weight = Slider(slider_ax_alignment, 'Alignment', 0, 5, valinit=1)
cohesion_weight = Slider(slider_ax_cohesion, 'Cohesion', 0, 5, valinit=1)
wall_avoidance_weight = Slider(slider_ax_wall, 'Wall Avoidance', 0, 5, valinit=2)

# Set up the simulation
num_boids = 15
flock = Flock(num_boids)
scatter = ax.scatter([b.position[0] for b in flock.boids], 
                     [b.position[1] for b in flock.boids])

def update(frame):

    flock.apply_behavior(0.1)
    positions = np.array([b.position for b in flock.boids])
    scatter.set_offsets(positions)
    
    # Print debug information
    print(f"Frame: {frame}, Num boids: {len(positions)}, X range: {positions[:,0].min():.2f}-{positions[:,0].max():.2f}, Y range: {positions[:,1].min():.2f}-{positions[:,1].max():.2f}")
    
    return scatter,

def reset(event):
    global flock
    flock = Flock(num_boids)

# Create a reset button
reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset', color=slider_color, hovercolor='0.975')
reset_button.on_clicked(reset)

# Create the animation
anim = FuncAnimation(fig, update, frames=None, interval=50, blit=True)

plt.show()