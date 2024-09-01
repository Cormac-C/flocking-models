import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Flock:
    def __init__(self, num_agents, dim=2):
        self.num_agents = num_agents
        self.dim = dim
        self.positions = np.random.rand(num_agents, dim) * 10
        self.velocities = np.random.randn(num_agents, dim)

    def update(self, dt):
        # Cucker-Smale flocking model
        for i in range(self.num_agents):
            alignment = np.zeros(self.dim)
            for j in range(self.num_agents):
                if i != j:
                    distance = np.linalg.norm(self.positions[i] - self.positions[j])
                    weight = 1 / (1 + distance**2)  # Communication weight
                    alignment += weight * (self.velocities[j] - self.velocities[i])
            
            self.velocities[i] += alignment * dt
        
        # Update positions
        self.positions += self.velocities * dt

        # Wrap around boundaries
        self.positions %= 10

# Set up the simulation
num_agents = 50
flock = Flock(num_agents)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
scatter = ax.scatter(flock.positions[:, 0], flock.positions[:, 1])

# Animation update function
def update(frame):
    flock.update(0.1)
    scatter.set_offsets(flock.positions)
    return scatter,

# Create the animation
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.title("Cucker-Smale Flocking Model")
plt.close()  # Prevent displaying the plot immediately

# Save the animation as a gif
anim.save('cucker_smale_flocking.gif', writer='pillow', fps=30)

print("Animation saved as 'cucker_smale_flocking.gif'")