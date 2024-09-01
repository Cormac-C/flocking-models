import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CONTAINER_SIZE = 30
MAX_SPEED = 4

class Boid:
    def __init__(self, x, y, vx, vy):
        self.position = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.acceleration = np.zeros(2)

    def update(self, dt):
        self.velocity += self.acceleration * dt
        self.velocity = np.clip(self.velocity, -MAX_SPEED, MAX_SPEED)  # Limit speed
        self.position += self.velocity * dt
        self.acceleration = np.zeros(2)

    def avoid_walls(self, margin=1):
        desired = None
        if self.position[0] < margin:
            desired = np.array([2, self.velocity[1]])
        elif self.position[0] > CONTAINER_SIZE - margin:
            desired = np.array([-2, self.velocity[1]])
        
        if self.position[1] < margin:
            desired = np.array([self.velocity[0], 2])
        elif self.position[1] > CONTAINER_SIZE - margin:
            desired = np.array([self.velocity[0], -2])
        
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
            separation = self.separation(boid)
            alignment = self.alignment(boid)
            cohesion = self.cohesion(boid)
            wall_avoidance = boid.avoid_walls()

            # Relative weights of behaviors
            separation_weight = 1
            alignment_weight = 1
            cohesion_weight = 1
            wall_avoidance_weight = 2

            # Transform to normalized vectors
            total_weight = separation_weight + alignment_weight + cohesion_weight + wall_avoidance_weight
            separation_weight /= total_weight
            alignment_weight /= total_weight
            cohesion_weight /= total_weight
            wall_avoidance_weight /= total_weight
            

            # Overall acceleration weight
            acceleration_weight = 5

            # Update acceleration
            boid.acceleration = acceleration_weight * (separation * separation_weight + alignment * alignment_weight + cohesion * cohesion_weight + wall_avoidance * wall_avoidance_weight)
            boid.update(dt)
            
            # Ensure boids stay within boundaries
            boid.position = np.clip(boid.position, 0, CONTAINER_SIZE)

    def separation(self, boid):
        steering = np.zeros(2)
        for other in self.boids:
            if other != boid:
                diff = boid.position - other.position
                dist = np.linalg.norm(diff)
                if 0 < dist < 1:
                    steering += diff / (dist * dist)
        return steering

    def alignment(self, boid):
        steering = np.zeros(2)
        total = 0
        for other in self.boids:
            if other != boid:
                if np.linalg.norm(boid.position - other.position) < 2:
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
                if np.linalg.norm(boid.position - other.position) < 2:
                    steering += other.position
                    total += 1
        if total > 0:
            steering /= total
            return (steering - boid.position) / 100
        return steering

# Set up the simulation
num_boids = 50
flock = Flock(num_boids)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, CONTAINER_SIZE)
ax.set_ylim(0, CONTAINER_SIZE)
scatter = ax.scatter([b.position[0] for b in flock.boids], 
                     [b.position[1] for b in flock.boids])

# Animation update function
def update(frame):
    flock.apply_behavior(0.1)
    scatter.set_offsets([b.position for b in flock.boids])
    return scatter,

# Create the animation
anim = FuncAnimation(fig, update, frames=300, interval=50, blit=True)

plt.title(f"Reynolds Boids Flocking Model (Container Size: {CONTAINER_SIZE}x{CONTAINER_SIZE})")
plt.close()  # Prevent displaying the plot immediately

# Save the animation as a gif
anim.save(f'reynolds_boids_flocking_{CONTAINER_SIZE}x{CONTAINER_SIZE}.gif', writer='pillow', fps=30)

print(f"Animation saved as 'reynolds_boids_flocking_{CONTAINER_SIZE}x{CONTAINER_SIZE}.gif'")