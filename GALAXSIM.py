import pygame
import math
from tqdm import tqdm
import numpy as np
from numba import cuda
import random as rnd
import conversion as conv

scale = 1.5e18
camera_x = 0
camera_y = 0

# Constants
SCREEN_WIDTH = 2200
SCREEN_HEIGHT = 1100
G = 6.67430e-11  # Gravitational constant
DT = 86000*365*25000000*4 # Time step (in seconds)
real_radius = 12.264e9

d = 5e20
m = 1.972e30*4.3e6

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ATTRACTOR_COLOR = (255, 0, 0)  # Color for main attractors
OTHER_COLOR = (0, 0, 255)       # Color for other bodies

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("N-body Simulation with Real Initial Conditions")
clock = pygame.time.Clock()

# Font
font = pygame.font.SysFont(None, 30)

class Body:
    def __init__(self, mass, pos, vel, color, radius, real_radius):
        self.mass = mass
        self.pos = pos
        self.vel = vel
        self.color = color
        self.radius = radius
        self.real_radius = real_radius
    
    def vel_to_color(self,velocity):
        x = velocity[0]
        y = velocity[1]
        speed = np.sqrt(x**2 + y**2)
        
        # Normalize speed to a value between 0 and 1
        normalized_speed = speed/2500 # You need to define max_speed or estimate it
        
        # Interpolate between blue (slow), green, and red (fast) based on normalized speed
        if normalized_speed > 1:
            normalized_speed = 1
        red = int(255 * normalized_speed)
        green = int(255 * (1 - normalized_speed))
        blue = 0  # No blue component
        
        return (red, green, blue)
             
    def draw(self):
        offset_x = SCREEN_WIDTH // 2 - camera_x
        offset_y = SCREEN_HEIGHT // 2 - camera_y
        pygame.draw.rect(screen, self.vel_to_color(self.vel), (self.pos[0] / scale + offset_x, self.pos[1] / scale + offset_y, self.radius, self.radius))
    
    

# CUDA kernel for updating body positions and velocities
@cuda.jit
def update_kernel(bodies_pos, bodies_vel, masses, dt, new_positions, new_velocities):
    i = cuda.grid(1)

    if i < len(bodies_pos):
        force_x = 0.0
        force_y = 0.0

        for j in range(len(bodies_pos)):
            if i != j:
                dx = bodies_pos[j][0] - bodies_pos[i][0]
                dy = bodies_pos[j][1] - bodies_pos[i][1]
                dist_sq = dx ** 2 + dy ** 2
                dist = math.sqrt(dist_sq)

                force_magnitude = G * masses[i] * masses[j] / dist_sq
                force_x += force_magnitude * dx / dist
                force_y += force_magnitude * dy / dist

        new_velocities[i][0] = bodies_vel[i][0] + force_x / masses[i] * dt
        new_velocities[i][1] = bodies_vel[i][1] + force_y / masses[i] * dt

        new_positions[i][0] = bodies_pos[i][0] + new_velocities[i][0] * dt
        new_positions[i][1] = bodies_pos[i][1] + new_velocities[i][1] * dt

def update(bodies, dt):
    num_bodies = len(bodies)

    # Copy data to device
    bodies_pos = np.array([body.pos for body in bodies], dtype=np.float64)
    bodies_vel = np.array([body.vel for body in bodies], dtype=np.float64)
    masses = np.array([body.mass for body in bodies], dtype=np.float64)

    new_positions = np.empty_like(bodies_pos)
    new_velocities = np.empty_like(bodies_vel)

    # Configure kernel and launch
    threads_per_block = 1024
    blocks_per_grid = (num_bodies + (threads_per_block - 1)) // threads_per_block

    update_kernel[blocks_per_grid, threads_per_block](bodies_pos, bodies_vel, masses, dt, new_positions, new_velocities)

    # Copy data back to host
    for i, body in enumerate(bodies):
        body.pos = new_positions[i]
        body.vel = new_velocities[i]
        

def create_galaxy(start_center,end_center,xrange,center_mass,camx,camy):
    
    mouse_x, mouse_y = start_center

    # Convert mouse position to world coordinates
    world_x = (mouse_x - SCREEN_WIDTH // 2 + camx) * scale 
    world_y = (mouse_y - SCREEN_HEIGHT // 2 + camy) * scale 
    
    # Calculate the velocity of the new body based on mouse drag
    dx = -(end_center[0] - start_center[0]) *5
    dy = -(end_center[1] - start_center[1]) *5
    vel = [dx, dy]
    
    new_body = Body(center_mass, [world_x, world_y], vel, "green", 3, real_radius)
    bodies.append(new_body)
    
    for i in tqdm(range(5000)):
        
        randomness = 0.01
        x = rnd.uniform(-xrange,xrange)
        y = rnd.uniform(-xrange,xrange)
        if np.sqrt(x**2 + y**2) < xrange:
            
            # Calculate the radius and angle
            r = np.sqrt(x**2 + y**2)
            
            theta = np.arctan(y/x)
            
            # Define the linear velocity (you may replace this with your actual value)
            v = np.sqrt(G*center_mass/r)
            if x <= 0:
                if r >= d*0.3 and r <= d :            #Corps moins massif , Couronne
                    if i % 5 == 0:
                        
                        vx = -v * np.cos(theta+conv.deg_to_rad(90)) 
                        vy = -v * np.sin(theta+conv.deg_to_rad(90)) 
                    
                        vx -= rnd.uniform(0,randomness) * vx - vel[0]
                        vy -= rnd.uniform(0,randomness) * vy - vel[1]
                        
                        new_body = Body(rnd.uniform(1e29,1e30), [x+world_x, y + world_y], [vx,vy], "lightblue", 1,real_radius)
                        bodies.append(new_body)
                else :                           #Corps plus massif , centre
                    if r <= d*0.3 and r >= d*0.2:
                        vx = -v * np.cos(theta+conv.deg_to_rad(90)) 
                        vy = -v * np.sin(theta+conv.deg_to_rad(90)) 
                        
                        vx -= rnd.uniform(0,randomness) * vx - vel[0]
                        vy -= rnd.uniform(0,randomness) * vy - vel[1]
                        
                        new_body = Body(rnd.uniform(1e30,1e32), [x+world_x, y + world_y], [vx,vy], "red", 1,real_radius)
                        bodies.append(new_body)
            else : 
                if r >= d*0.3 and r <= d :       #Corps moins massif , Couronne
                    if i % 5 == 0:
                        vx = v * np.cos(theta+conv.deg_to_rad(90)) 
                        vy = v * np.sin(theta+conv.deg_to_rad(90)) 
                        
                        vx -= rnd.uniform(0,randomness) * vx - vel[0]
                        vy -= rnd.uniform(0,randomness) * vy - vel[1] 
                        
                        new_body = Body(rnd.uniform(1e29,1e30), [x+world_x, y + world_y], [vx,vy], "lightblue", 1,real_radius)
                        bodies.append(new_body)
                else :                          #Corps plus massif , centre
                    if r <= d*0.3 and r >= d*0.2:
                        vx = v * np.cos(theta+conv.deg_to_rad(90)) 
                        vy = v * np.sin(theta+conv.deg_to_rad(90)) 
                        
                        vx -= rnd.uniform(0,randomness) * vx - vel[0] 
                        vy -= rnd.uniform(0,randomness) * vy - vel[1]
                        
                        new_body = Body(rnd.uniform(1e30,1e32), [x+world_x, y + world_y], [vx,vy], "red", 1,real_radius)
                        bodies.append(new_body)

def place_star(start_center,end_center,mass):                  
    
    mouse_x, mouse_y = start_pos

    # Convert mouse position to world coordinates
    world_x = (mouse_x - SCREEN_WIDTH // 2) * scale
    world_y = (mouse_y - SCREEN_HEIGHT // 2) * scale

    

    # Calculate the velocity of the new body based on mouse drag
    dx = -(end_pos[0] - start_pos[0]) * 5
    dy = -(end_pos[1] - start_pos[1]) * 5
    new_vel = [dx, dy]

    # Create a new body and add it to the list
    new_body = Body(mass, [world_x, world_y], new_vel, "magenta", 2, real_radius)
    bodies.append(new_body)


bodies_data = [
    
    {"mass": 1, "pos": [0, 0], "vel": [1, 1], "color": "cyan",'radius': 0, "real_radius": 0},
    
    
]

# Create bodies
bodies = []
for data in bodies_data:
    body = Body(data["mass"], data["pos"], data["vel"], data["color"], data["radius"], data["real_radius"])
    bodies.append(body)

# Main loop
running = True
add_body = False
add_galaxy = False
start_pos = None

while running:
    screen.fill(BLACK)

    # Get the current mouse position
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left-click
                add_body = True
                start_pos = event.pos
                print("Mouse button down:", start_pos)
            elif event.button == 3:  # Right-click
                add_galaxy = True
                start_pos = event.pos
                print("Mouse button down:", start_pos)
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if add_body:
                    end_pos = event.pos                  
                    # Define the mass of the new body
                    mass = 6e33
                    
                    place_star(start_pos,end_pos,mass)                 
                    print("New Star added")
        
                    add_body = False
                
            elif event.button == 3:
                if add_galaxy:
                    end_pos = event.pos
                    # Define the mass of the massive body at the center of teh galaxy
                    center_mass = m
                    # Define the radius of the Galaxy (d = 52 800 light-years here)
                    xrange = d
                    create_galaxy(start_pos,end_pos, xrange ,center_mass,camera_x,camera_y)         
                    print("New Galaxy added")
                
                add_galaxy = False
                
        
        
        elif event.type == pygame.KEYDOWN:
            # Move camera based on arrow keys
            if event.key == pygame.K_LEFT:
                camera_x -= 100
            elif event.key == pygame.K_RIGHT:
                camera_x += 100
            elif event.key == pygame.K_UP:
                camera_y -= 100
            elif event.key == pygame.K_DOWN:
                camera_y += 100
    
    

    if add_body:
        # Draw a line from the start position to the current mouse position
        dx = mouse_pos[0] - start_pos[0]
        dy = mouse_pos[1] - start_pos[1]
        pygame.draw.line(screen, WHITE, start_pos, mouse_pos, 1)
        
    if add_galaxy:
        # Draw a line from the start position to the current mouse position
        dx = mouse_pos[0] - start_pos[0]
        dy = mouse_pos[1] - start_pos[1]
        pygame.draw.line(screen, WHITE, start_pos, mouse_pos, 1)
        
    
    # Update and draw bodies
    update(bodies, DT)
    for body in bodies:
        body.draw()
    
    # Display FPS
    fps_text = font.render("FPS: {:.2f}".format(clock.get_fps()), True, WHITE)
    screen.blit(fps_text, (10, 10))

    pygame.display.flip()
    clock.tick(1000)

pygame.quit()












