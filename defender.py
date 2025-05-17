import pygame
import sys
import math
import random
import numpy # For sound generation
import time # For respawn delay

# Initialize Pygame & Mixer
pygame.mixer.pre_init(44100, -16, 1, 512) # Frequency, size, channels, buffer
pygame.init()
pygame.mixer.init() # Initialize mixer

# Screen dimensions (16:9 ratio)
ASPECT_RATIO = 16 / 9
HEIGHT = 600
WIDTH = int(HEIGHT * ASPECT_RATIO) # Calculate width for 16:9 (1066)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vector Defender - Continuous!")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
STAR_COLOR = (200, 200, 200)
LASER_COLOR = (0, 255, 255) # Cyan laser for better visibility
MINIMAP_BG = (30, 30, 30, 200) # Semi-transparent dark grey
MINIMAP_BORDER = (100, 100, 100)

# Game clock for controlling frame rate
clock = pygame.time.Clock()
FPS = 60

# Game states
GAME_STATE_START = "start"
GAME_STATE_PLAYING = "playing"
GAME_STATE_LEVEL_TRANSITION = "level_transition"
GAME_STATE_PLAYER_DIED = "player_died" # State for respawn delay
GAME_STATE_GAME_OVER = "game_over"
game_state = GAME_STATE_START

# Scoring, lives, and level
score = 0
level_score = 0 # Score accumulated during the current level
initial_humanoids_count = 0 # Track humanoids at level start
lives = 3
level = 1
font = pygame.font.Font(None, 30)
large_font = pygame.font.Font(None, 74)
info_font = pygame.font.Font(None, 36)

# World dimensions (Adjusted based on new WIDTH)
WORLD_WIDTH_MULTIPLIER = 3
WORLD_WIDTH = WIDTH * WORLD_WIDTH_MULTIPLIER # 1066 * 3 = 3198

# Camera offset
camera_x = 0

# Timers
level_transition_timer = 0
LEVEL_TRANSITION_DURATION = 1500 # Milliseconds
respawn_timer = 0
RESPAWN_DELAY = 2000 # Milliseconds (2 seconds)

# Mini-map settings (Adjusted X position based on new WIDTH)
MINIMAP_WIDTH = 150
MINIMAP_HEIGHT = 50
MINIMAP_X = WIDTH - MINIMAP_WIDTH - 10 # Position relative to new width
MINIMAP_Y = 10
MINIMAP_SCALE_X = MINIMAP_WIDTH / WORLD_WIDTH
MINIMAP_SCALE_Y = MINIMAP_HEIGHT / HEIGHT


# --- Sound Generation ---
SAMPLE_RATE = 44100

def generate_sound_array(freq, duration_ms, vol=0.1, decay=True):
    """ Generates a numpy array for a sine wave sound. """
    t = numpy.linspace(0., duration_ms / 1000., int(SAMPLE_RATE * duration_ms / 1000.), endpoint=False)
    wave = numpy.sin(2. * numpy.pi * freq * t)
    if decay:
        decay_rate = 5.0 # Controls how fast the sound decays
        decay_factor = numpy.exp(-decay_rate * t)
        wave *= decay_factor
    wave *= vol * 32767 # Scale to 16-bit integer range
    return wave.astype(numpy.int16)

def generate_explosion_array(duration_ms, vol=0.2):
    """ Generates a numpy array for a simple noise explosion. """
    num_samples = int(SAMPLE_RATE * duration_ms / 1000.)
    noise = numpy.random.uniform(-1, 1, num_samples)
    # Apply a decay envelope
    t = numpy.linspace(0., duration_ms / 1000., num_samples, endpoint=False)
    decay_rate = 8.0
    decay_factor = numpy.exp(-decay_rate * t)
    noise *= decay_factor
    noise *= vol * 32767
    return noise.astype(numpy.int16)

# Create Sound Objects
try:
    pew_sound_arr = generate_sound_array(880, 100, vol=0.08, decay=True) # Higher pitch pew
    pew_sound = pygame.mixer.Sound(buffer=pew_sound_arr)

    explosion_sound_arr = generate_explosion_array(400, vol=0.15) # Longer, louder explosion
    explosion_sound = pygame.mixer.Sound(buffer=explosion_sound_arr)
    sounds_loaded = True
except Exception as e:
    print(f"Error initializing sounds: {e}")
    sounds_loaded = False

def play_sound(sound):
    """ Safely plays a sound if sounds are loaded. """
    if sounds_loaded:
        sound.play()

# --- Base GameObject Class ---
class GameObject(pygame.sprite.Sprite):
    """Base class for all game objects with vector graphics."""
    def __init__(self, position, color, points=None):
        super().__init__()
        self.position = pygame.Vector2(position) # World position
        self.color = color
        self.base_points = points or []
        self.angle = 0
        self.velocity = pygame.Vector2(0, 0)

        if self.base_points:
            self.max_radius = max(math.hypot(p[0], p[1]) for p in self.base_points) if self.base_points else 1
        else: self.max_radius = 1

        # Keep image/rect for potential collision detection, though drawing uses points
        rect_size = self.max_radius * 2
        self.image = pygame.Surface((rect_size, rect_size), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(int(self.position.x), int(self.position.y)))

    def get_transformed_points(self):
        """ Calculates the points rotated and translated to world coordinates. """
        transformed_points = []
        if self.base_points:
            cos_a = math.cos(self.angle); sin_a = math.sin(self.angle)
            for point in self.base_points:
                rotated_x = point[0] * cos_a - point[1] * sin_a
                rotated_y = point[0] * sin_a + point[1] * cos_a
                world_x = self.position.x + rotated_x
                world_y = self.position.y + rotated_y
                transformed_points.append((world_x, world_y))
        return transformed_points

    def draw(self, surface, camera_offset_x):
        """Draws the object relative to the camera, handling world wrap."""
        world_points = self.get_transformed_points()
        if not world_points: return

        # --- SIMPLIFIED COORDINATE CALCULATION (FOR DEBUGGING) ---
        # This version removes the complex wrap adjustment (if dx > WORLD_WIDTH / 2...)
        screen_points = []
        for wx, wy in world_points:
            sx = wx - camera_offset_x # Simple camera offset only
            sy = wy
            screen_points.append((int(sx), int(sy)))
        # --- END OF SIMPLIFIED CALCULATION ---

        # Draw the primary instance (using the potentially incorrect screen_points for wrap cases)
        self._draw_polygon_if_visible(surface, screen_points)

        # --- Draw wrapped instances if the object might be crossing the wrap boundary ---
        # This logic might be less effective now because screen_points wasn't adjusted for wrap,
        # but we leave it for now to isolate the coordinate calculation change.
        world_x_coords = [p[0] for p in world_points]
        min_world_x = min(world_x_coords)
        max_world_x = max(world_x_coords)

        # Condition 1: Object is near the LEFT edge of the world...
        if min_world_x < self.max_radius * 2 and camera_offset_x > WORLD_WIDTH - WIDTH:
            # Create screen points shifted right by the world width
            # NOTE: This uses the UNADJUSTED screen_points from the simplified calculation above
            wrapped_screen_points = [(p[0] + WORLD_WIDTH, p[1]) for p in screen_points]
            self._draw_polygon_if_visible(surface, wrapped_screen_points)

        # Condition 2: Object is near the RIGHT edge of the world...
        elif max_world_x > WORLD_WIDTH - self.max_radius * 2 and camera_offset_x < WIDTH:
            # Create screen points shifted left by the world width
            # NOTE: This uses the UNADJUSTED screen_points from the simplified calculation above
            wrapped_screen_points = [(p[0] - WORLD_WIDTH, p[1]) for p in screen_points]
            self._draw_polygon_if_visible(surface, wrapped_screen_points)


    def _draw_polygon_if_visible(self, surface, screen_points):
        """ Helper function to draw polygon only if its bounding box overlaps the screen. """
        line_width = 2 if isinstance(self, Laser) else 1 # Player laser is thicker

        if not screen_points or len(screen_points) < 2: return

        min_x = min(p[0] for p in screen_points)
        max_x = max(p[0] for p in screen_points)
        min_y = min(p[1] for p in screen_points)
        max_y = max(p[1] for p in screen_points)

        # Corrected Check: Draw only if the bounding box overlaps the screen rectangle
        if max_x > 0 and min_x < WIDTH and max_y > 0 and min_y < HEIGHT:
            try:
                if len(screen_points) >= 3:
                    pygame.draw.polygon(surface, self.color, screen_points, line_width)
                elif len(screen_points) == 2: # Primarily for lasers
                    pygame.draw.line(surface, self.color, screen_points[0], screen_points[1], line_width)
            except ValueError:
                # Silently ignore ValueErrors which might occur with weird coordinates
                pass

    def update(self):
        """Updates world position, handles world wrap, and updates rect."""
        self.position += self.velocity
        # World wrap for horizontal position
        self.position.x %= WORLD_WIDTH
        # Update rect center for collision detection (uses world coordinates)
        self.rect.center = (int(self.position.x), int(self.position.y))

    def destroy(self, release_humanoid=True): # Add flag to control humanoid release
        """ Creates explosion effect, plays sound, kills object, and potentially releases humanoid. """
        play_sound(explosion_sound)
        num_particles = random.randint(15, 25)
        for _ in range(num_particles):
            # Particles spawn at the object's current position
            particle = ExplosionParticle(self.position.copy(), self.color)
            all_sprites.add(particle)
            explosions.add(particle) # Add to explosion group for separate update/draw if needed

        # --- Release captured humanoid if this is a Lander ---
        if release_humanoid and isinstance(self, Lander) and self.target_humanoid and self.target_humanoid.alive() and self.target_humanoid.is_captured:
            print(f"Releasing humanoid from destroyed Lander {id(self)}")
            self.target_humanoid.is_captured = False
            self.target_humanoid.is_falling = True
            self.target_humanoid = None # Clear reference so Lander doesn't track it

        self.kill() # Remove this sprite from all groups


# --- Player Class ---
class Player(GameObject):
    """Represents the player ship."""
    def __init__(self, position):
        # Points define the ship shape relative to its center (0,0)
        self.points_right = [ (15, 0), (-12, -8), (-7, 0), (-12, 8) ] # Facing right
        self.points_left = [(-p[0], p[1]) for p in self.points_right] # Facing left (mirrored)
        super().__init__(position, GREEN, self.points_right)
        self.facing_direction = 1 # 1 for right, -1 for left
        self.base_points = self.points_right # Start facing right
        self.acceleration = 0.28; self.friction = 0.95; self.max_speed = 7
        self.fire_cooldown = 180 # Milliseconds
        self.can_fire = True; self.fire_cooldown_timer = 0
        self.accel = pygame.Vector2(0, 0) # Acceleration vector for input
        self.is_destroyed = False
        self.invulnerable = False # For brief period after respawn
        self.invulnerable_timer = 0
        self.INVULNERABLE_DURATION = 1500 # Milliseconds
        self.visible = True # For flashing effect when invulnerable
        self.approx_length = 27 # Approximate length (15 - (-12)) for laser range calc

    def handle_input(self, keys):
        if self.is_destroyed: return
        self.accel.x = 0; self.accel.y = 0
        new_facing_direction = self.facing_direction
        if keys[pygame.K_LEFT]: self.accel.x = -self.acceleration; new_facing_direction = -1
        if keys[pygame.K_RIGHT]: self.accel.x = self.acceleration; new_facing_direction = 1
        if keys[pygame.K_UP]: self.accel.y = -self.acceleration
        if keys[pygame.K_DOWN]: self.accel.y = self.acceleration

        # Update ship points if direction changes
        if new_facing_direction != self.facing_direction:
            self.facing_direction = new_facing_direction
            self.base_points = self.points_left if self.facing_direction == -1 else self.points_right
            # Recalculate max_radius based on new points
            self.max_radius = max(math.hypot(p[0], p[1]) for p in self.base_points) if self.base_points else 1

        if keys[pygame.K_SPACE] and self.can_fire:
             self.fire(); self.can_fire = False; self.fire_cooldown_timer = pygame.time.get_ticks()

    def fire(self):
        play_sound(pew_sound)
        offset_magnitude = 18 # Fire from slightly ahead of the nose
        # Calculate offset based on facing direction
        laser_start_offset = pygame.Vector2(offset_magnitude * self.facing_direction, 0)
        laser_position = (self.position + laser_start_offset)
        laser_position.x %= WORLD_WIDTH # Wrap laser start position if needed

        laser_speed = 14
        # Laser velocity includes player's velocity slightly + base speed
        base_laser_velocity = pygame.Vector2(laser_speed * self.facing_direction, 0)
        laser_velocity = base_laser_velocity + self.velocity * 0.5 # Inherit some ship momentum
        # Pass approximate player length to Laser for range calculation
        laser = Laser(laser_position, LASER_COLOR, laser_velocity, self.approx_length)
        all_sprites.add(laser); lasers.add(laser) # Add to groups

    def update(self):
        current_time = pygame.time.get_ticks()
        if self.is_destroyed: return

        # Invulnerability logic
        if self.invulnerable:
            if current_time - self.invulnerable_timer > self.INVULNERABLE_DURATION:
                self.invulnerable = False; self.visible = True # Become solid and vulnerable
            else:
                # Flash visibility
                self.visible = (current_time // 100) % 2 == 0

        # Apply acceleration and friction
        self.velocity += self.accel; self.velocity *= self.friction
        # Stop if speed is negligible
        if self.velocity.length_squared() < 0.01: self.velocity.xy = (0, 0)
        # Clamp speed to max_speed
        if self.velocity.length_squared() > self.max_speed**2: self.velocity.scale_to_length(self.max_speed)

        super().update() # Handles position += velocity and world wrap (from GameObject)

        # Vertical Boundary Checks (Stay within screen height)
        if self.position.y > HEIGHT - self.max_radius: self.position.y = HEIGHT - self.max_radius; self.velocity.y = 0
        elif self.position.y < self.max_radius: self.position.y = self.max_radius; self.velocity.y = 0

        # Cooldown timer for firing
        if not self.can_fire and current_time - self.fire_cooldown_timer > self.fire_cooldown:
             self.can_fire = True

    def crash(self):
        global lives, game_state, respawn_timer
        if not self.is_destroyed and not self.invulnerable:
            print("Player crashed!")
            self.is_destroyed = True; lives -= 1
            # Call destroy WITHOUT releasing humanoid (player doesn't carry)
            self.destroy(release_humanoid=False)
            if lives > 0:
                game_state = GAME_STATE_PLAYER_DIED; respawn_timer = pygame.time.get_ticks()
            else:
                # Use a timer event to delay game over slightly, allows explosion to show
                pygame.time.set_timer(pygame.USEREVENT_GAME_OVER, 1000, 1) # 1000ms, runs once

    def respawn(self, respawn_pos):
        print("Respawning player...")
        self.position = pygame.Vector2(respawn_pos); self.velocity = pygame.Vector2(0, 0)
        self.is_destroyed = False; self.can_fire = True; self.facing_direction = 1
        self.base_points = self.points_right # Ensure facing right on respawn
        self.invulnerable = True
        self.invulnerable_timer = pygame.time.get_ticks(); self.visible = True
        # Re-add to sprite group if it was removed (e.g., by self.kill in crash)
        if self not in all_sprites: all_sprites.add(self)

    def draw(self, surface, camera_offset_x):
        # Only draw if visible (handles flashing during invulnerability)
        if self.visible:
            super().draw(surface, camera_offset_x) # Use GameObject's draw


# --- Explosion Particle Class ---
class ExplosionParticle(GameObject):
    """ A single particle for the explosion effect. """
    def __init__(self, position, base_color):
        # Particles don't use predefined points, they are drawn as circles
        super().__init__(position, base_color, points=None)
        speed = random.uniform(2, 6); angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifetime = random.randint(20, 40) # Frames
        self.start_radius = random.randint(2, 4)
        self.radius = self.start_radius
        # Randomize particle color within explosion palette
        self.color = random.choice([RED, ORANGE, YELLOW, WHITE])

    def update(self):
        super().update() # Applies velocity and world wrap
        self.lifetime -= 1
        # Shrink particle over time
        self.radius = self.start_radius * (self.lifetime / 40.0)
        if self.lifetime <= 0 or self.radius < 1:
            self.kill() # Remove particle

    def draw(self, surface, camera_offset_x):
        # Explosion particles override the default draw because they are circles
        # Calculate screen pos relative to camera, handling wrap
        dx = self.position.x - camera_offset_x
        if dx > WORLD_WIDTH / 2: dx -= WORLD_WIDTH
        elif dx < -WORLD_WIDTH / 2: dx += WORLD_WIDTH
        screen_x = int(dx) # Position relative to camera's left edge
        screen_y = int(self.position.y)

        # Corrected visibility check for particles
        min_x, max_x = screen_x - self.radius, screen_x + self.radius
        min_y, max_y = screen_y - self.radius, screen_y + self.radius
        # Use the same logic as _draw_polygon_if_visible for consistency
        if max_x > 0 and min_x < WIDTH and max_y > 0 and min_y < HEIGHT:
            if self.radius >= 1: # Don't draw tiny circles
                pygame.draw.circle(surface, self.color, (screen_x, screen_y), int(self.radius))


# --- Laser Class ---
class Laser(GameObject):
    """Represents a laser projectile fired by the player."""
    # MAX_RANGE constant removed, lifetime calculated instead

    def __init__(self, position, color, velocity, player_length=27): # Get player length for range
        # Laser points define a short horizontal line segment
        points = [(0, 0), (12, 0)]; # Make laser slightly longer for visibility
        super().__init__(position, color, points)
        self.velocity = velocity
        self.start_position = position.copy() # Store starting position (not used currently)

        # Calculate range based on player length
        max_distance = 10 * player_length # Approx 10 ship lengths

        # Calculate lifetime based on speed and max range
        speed = velocity.length()
        if speed > 0:
            # Lifetime in frames = (Distance / Speed) * FPS
            self.lifetime = int((max_distance / speed) * FPS)
        else:
            self.lifetime = 1 # Remove immediately if no speed

        # Angle based on velocity (mostly horizontal for player)
        if velocity.length_squared() > 0: self.angle = math.atan2(velocity.y, velocity.x)
        else: self.angle = 0 # Default angle if no velocity

    def update(self, terrain_obj): # Add terrain_obj argument for collision
        """ Update laser, check range (lifetime) and terrain collision. """
        prev_pos = self.position.copy() # Store position before update
        super().update() # Applies velocity and world wrap
        self.lifetime -= 1

        # Kill if lifetime expires
        if self.lifetime <= 0:
            self.kill()
            return

        # --- Terrain Collision Check ---
        # Get terrain height at current laser position
        current_terrain_y = terrain_obj.get_height_at(self.position.x)
        # Simple check: kill if laser is below terrain
        if self.position.y > current_terrain_y:
            self.kill()
            # Could add a small explosion/spark effect here


# --- Terrain Class ---
class Terrain:
    """Generates and draws the scrolling, varied terrain."""
    def __init__(self, world_width, screen_height, segment_length=25):
        self.world_width = world_width; self.screen_height = screen_height
        self.segment_length = segment_length;
        self.points = self._generate_terrain()

    def _generate_terrain(self):
        """ Creates the list of (x, y) points defining the terrain surface. """
        points = []; x = 0; y = self.screen_height * 0.80 # Start height
        min_y = self.screen_height * 0.50; max_y = self.screen_height - 60 # Height bounds
        slope = 0 # Current slope tendency
        while x < self.world_width:
            points.append((x, int(y)))
            # Randomly decide how the terrain changes
            change_type = random.random()
            if change_type < 0.05: # Large sudden jump
                max_deviation = 100; dy = random.randint(-max_deviation, max_deviation); slope = 0
            elif change_type < 0.15: # Flat segment
                 dy = 0; slope = 0
            elif change_type < 0.5: # Moderate slope change
                max_deviation = 40; slope_change = random.randint(-max_deviation // 2, max_deviation // 2)
                # Tendency to flatten out
                if slope > 0: slope = max(-max_deviation, slope - slope_change - 5)
                elif slope < 0: slope = min(max_deviation, slope + slope_change + 5)
                else: slope += slope_change
                dy = slope
            else: # Small random variation on current slope
                max_deviation = 40; dy = slope + random.randint(-max_deviation // 4, max_deviation // 4)

            y += dy;
            y += 0.1 # Gentle downward bias? (negligible effect)
            y = max(min_y, min(y, max_y)); # Clamp y within bounds
            x += self.segment_length # Move to next point

        # Add a final point that wraps back to the start height for seamless looping
        # Ensure the x-coordinate is exactly the world width for calculations
        points.append((self.world_width, points[0][1]))
        print(f"Generated {len(points)} varied terrain points up to x={points[-1][0]}")
        return points

    def get_height_at(self, world_x):
        """ Interpolates the terrain height at a specific world x-coordinate. """
        # Ensure world_x is within bounds using modulo
        world_x %= self.world_width

        # Find the two terrain points surrounding the world_x
        index = int(world_x // self.segment_length)
        # Clamp index to prevent out-of-bounds errors (especially near wrap point)
        index = max(0, min(index, len(self.points) - 2))
        p1 = self.points[index]; p2 = self.points[index + 1]

        # Handle vertical terrain segments (shouldn't happen with generation logic)
        if p2[0] == p1[0]: return p1[1]

        # Calculate the width of the segment between p1 and p2
        segment_width = p2[0] - p1[0]

        if segment_width <= 0: segment_width = self.segment_length # Fallback

        # Calculate 't', the fractional distance of world_x between p1[0] and p2[0]
        local_x = world_x - p1[0] # x relative to the start of the segment
        t = local_x / segment_width

        # Clamp t between 0 and 1 to avoid extrapolation errors
        t = max(0.0, min(1.0, t))

        # Linear interpolation: height = y1 + t * (y2 - y1)
        height = p1[1] + t * (p2[1] - p1[1])
        return int(height)


    def draw(self, surface, camera_offset_x):
        """ Draws the terrain line and fills below, handling wrap correctly. """
        # Determine how many segments are potentially visible on screen (+ buffer)
        num_segments_to_draw = int(WIDTH / self.segment_length) + 4
        # Find the world x-coordinate corresponding to the left edge of the camera
        start_world_x = camera_offset_x
        # Calculate the index of the first terrain point visible (or just off-screen left)
        start_index = int((start_world_x % WORLD_WIDTH) // self.segment_length)

        line_strip_points = [] # Points for the blue terrain outline
        fill_poly_points = [] # Points for the black filled area below terrain

        # --- Calculate points for drawing ---
        for i in range(num_segments_to_draw):
            # Get the index, wrapping around using modulo
            current_index = (start_index + i) % (len(self.points) - 1) # Use -1 because last point is wrap point
            world_x, world_y = self.points[current_index]

            # Calculate screen x relative to camera, handling world wrap explicitly
            screen_x = world_x - camera_offset_x
            # If the point is more than half a world away, assume it needs wrapping
            if abs(screen_x - WIDTH/2) > WORLD_WIDTH/2 : # Check distance from screen center
                 if screen_x > 0: screen_x -= WORLD_WIDTH # Point is far right, wrap it left
                 else: screen_x += WORLD_WIDTH # Point is far left, wrap it right

            point = (int(screen_x), int(world_y))
            # Add point to lists only if it's within a generous horizontal range of the screen
            # This prevents processing points way off screen
            if -WIDTH < point[0] < WIDTH * 2 :
                 line_strip_points.append(point)
                 fill_poly_points.append(point)

        # --- Draw the terrain ---
        if len(line_strip_points) >= 2:
            # Prepare fill polygon points
            # We need to add bottom-left and bottom-right corners to close the polygon shape for filling
            # Sort points by x-coordinate to ensure correct order for polygon drawing
            drawable_fill_points = sorted(fill_poly_points, key=lambda p: p[0])
            drawable_line_points = sorted(line_strip_points, key=lambda p: p[0]) # Also sort line points

            if drawable_fill_points: # Ensure we have points to draw
                # Define bottom corners based on the leftmost and rightmost visible terrain points
                bottom_left = (drawable_fill_points[0][0], HEIGHT)
                bottom_right = (drawable_fill_points[-1][0], HEIGHT)
                # Construct the final list for the fill polygon:
                # [bottom-left, terrain_point1, terrain_point2, ..., terrain_pointN, bottom-right]
                final_fill_points = [bottom_left] + drawable_fill_points + [bottom_right]

                # Draw the filled polygon (black area below terrain)
                if len(final_fill_points) >= 3:
                    try:
                         pygame.draw.polygon(surface, BLACK, final_fill_points, 0) # 0 width = fill
                    except ValueError: # Handle potential errors with coordinate lists
                         pass # Silently ignore if points are bad

            # Draw the terrain surface line (blue outline)
            if len(drawable_line_points) >= 2:
                 try:
                     # Use draw.lines for connected segments, False means not closed loop
                     pygame.draw.lines(surface, BLUE, False, drawable_line_points, 2) # Thickness 2
                 except ValueError:
                     pass # Silently ignore

# --- ParallaxLayer Class ---
class ParallaxLayer:
    """ Handles parallax stars, respecting world wrap. """
    def __init__(self, world_width, num_elements, color, size_range, scroll_factor):
        self.elements = []; self.color = color; self.scroll_factor = scroll_factor
        # Generate random stars (x, y, size) across the world
        for _ in range(num_elements):
            x = random.uniform(0, world_width); y = random.uniform(0, HEIGHT)
            size = random.randint(size_range[0], size_range[1])
            self.elements.append({'x': x, 'y': y, 'size': size}) # Store as dict

    def draw(self, surface, camera_offset_x):
        """ Draws stars, handling wrap. Terrain fill covers those below. """
        for element in self.elements:
            world_x, world_y, size = element['x'], element['y'], element['size']
            # Calculate screen x based on camera position and parallax scroll factor
            # Use modulo for seamless wrapping
            parallax_cam_x = camera_offset_x * self.scroll_factor
            screen_x = (world_x - parallax_cam_x) % WORLD_WIDTH

            # Further adjustment: if the calculated screen_x is far offscreen due to the modulo,
            # bring it back by adding/subtracting WORLD_WIDTH. This helps ensure stars near the
            # wrap point are drawn correctly when the camera is near the edge.
            if screen_x > WIDTH + size and world_x > parallax_cam_x :
                 screen_x -= WORLD_WIDTH # Wrapped from right edge, bring left
            elif screen_x < -size and world_x < parallax_cam_x:
                 screen_x += WORLD_WIDTH # Wrapped from left edge, bring right

            # Draw if horizontally within or near screen bounds
            # This simple check is likely sufficient for small stars
            if -size <= screen_x <= WIDTH + size:
                # Draw smaller stars as single pixels, larger as circles
                if size <= 1:
                    try:
                        # Use set_at for single pixels (can be slow if many)
                        surface.set_at((int(screen_x), int(world_y)), self.color)
                    except IndexError: pass # Ignore if calculated pos is outside surface bounds
                else:
                    pygame.draw.circle(surface, self.color, (int(screen_x), int(world_y)), size)


# --- Humanoid Class ---
class Humanoid(GameObject):
    """ Represents a humanoid on the terrain. """
    def __init__(self, position):
        # Simple stick figure points
        points = [ (0, -8), (0, 0), (-4, 5), (0, 0), (4, 5), (0, 0), (-5, -4), (0, 0), (5, -4) ]
        super().__init__(position, WHITE, points)
        self.is_captured = False; self.is_falling = False; self.fall_speed = 2.5;
        self.capture_target_y = -30 # Y-coord where humanoid is considered fully abducted off-screen

    def update(self, terrain_obj):
        """ Update humanoid state (on ground, falling, captured). """
        if self.is_captured:
            # Movement handled by Lander, just check for abduction
            if self.position.y < self.capture_target_y:
                print(f"Humanoid {id(self)} abducted!")
                self.kill() # Remove abducted humanoid
        elif self.is_falling:
            # Apply gravity (fall speed)
            self.velocity = pygame.Vector2(0, self.fall_speed)
            terrain_y = terrain_obj.get_height_at(self.position.x)
            # Check if landed on terrain
            if self.position.y >= terrain_y - self.max_radius: # Use max_radius as approx height
                self.position.y = terrain_y - self.max_radius; self.is_falling = False; self.velocity = pygame.Vector2(0, 0)
        else: # On ground
            self.velocity = pygame.Vector2(0, 0) # Stand still on ground
            # Ensure humanoid stays exactly on terrain surface
            terrain_y = terrain_obj.get_height_at(self.position.x)
            # Adjust position if slightly off the ground (e.g., due to terrain changes)
            if abs(self.position.y - (terrain_y - self.max_radius)) > 2:
                 self.position.y = terrain_y - self.max_radius

        super().update() # Applies velocity and world wrap

# --- Lander Class ---
class Lander(GameObject):
    """Represents an enemy Lander ship."""
    def __init__(self, position, speed_multiplier=1.0):
        # Points defining the Lander shape
        points = [ (-8, 8), (-10, 0), (-8, -8), (8, -8), (10, 0), (8, 8), (5, 8), (0, 12), (-5, 8) ] # Legs, body, top
        super().__init__(position, RED, points)
        self.state = "descending"; self.target_humanoid = None # AI state and target
        # Apply speed multiplier based on level
        self.seek_speed = 1.0 * speed_multiplier
        self.descent_speed = 0.5 * speed_multiplier
        self.capture_distance = 20 # Distance to start capturing
        # Fire cooldown adjusted by speed multiplier (faster landers fire faster)
        self.fire_timer = random.randint(int(90 / speed_multiplier), int(240 / speed_multiplier))
        self.fire_cooldown_base = (90, 240); # Base range in frames
        self.horizontal_drift_speed = 0.4 * speed_multiplier
        self.angle = random.uniform(-0.1, 0.1) # Slight random tilt

    def find_target(self, humanoids_group):
        """ Finds the closest available humanoid. """
        self.target_humanoid = None; min_distance_sq = float('inf') # Use infinity for initial check
        for humanoid in humanoids_group:
            # Target only alive, not captured, not falling humanoids
            if humanoid.alive() and not humanoid.is_captured and not humanoid.is_falling:
                # Calculate shortest distance considering world wrap
                dx = abs(self.position.x - humanoid.position.x)
                wrapped_dx = min(dx, WORLD_WIDTH - dx) # Horizontal distance via wrap or direct
                dy = abs(self.position.y - humanoid.position.y)
                distance_sq = wrapped_dx**2 + dy**2
                if distance_sq < min_distance_sq:
                     min_distance_sq = distance_sq; self.target_humanoid = humanoid

    def update(self, humanoids_group, terrain_obj, player_pos):
        """ Lander AI state machine update. """
        # --- State Machine Logic ---
        if self.state == "descending":
            self.velocity.y = self.descent_speed
            # Random horizontal drift while descending
            if abs(self.velocity.x) < self.horizontal_drift_speed * 0.5 and random.random() < 0.02:
                 self.velocity.x = random.uniform(-self.horizontal_drift_speed, self.horizontal_drift_speed)
            else: # Dampen drift
                 self.velocity.x *= 0.95

            terrain_height = terrain_obj.get_height_at(self.position.x)
            # Once low enough, try to find a target
            if self.position.y > terrain_height - HEIGHT * 0.6: # Check relative height
                self.find_target(humanoids_group)
                if self.target_humanoid:
                     self.state = "seeking"; self.velocity.x = 0 # Stop drifting, start seeking
                # If no target found and getting close to ground, hover
                elif self.position.y > terrain_height - self.max_radius * 3:
                     self.velocity.y = 0 # Stop descending
                     # Gentle hover drift if no target
                     if abs(self.velocity.x) < 0.1:
                         self.velocity.x = random.uniform(-self.horizontal_drift_speed*0.5, self.horizontal_drift_speed*0.5)

        elif self.state == "seeking":
            if self.target_humanoid and self.target_humanoid.alive():
                # Check if target is still available (not captured/falling)
                if not self.target_humanoid.is_captured and not self.target_humanoid.is_falling:
                    # Calculate vector to target, considering wrap
                    dx = self.target_humanoid.position.x - self.position.x
                    if dx > WORLD_WIDTH / 2: dx -= WORLD_WIDTH
                    elif dx < -WORLD_WIDTH / 2: dx += WORLD_WIDTH
                    dy = self.target_humanoid.position.y - self.position.y
                    direction = pygame.Vector2(dx, dy)
                    dist = direction.length()

                    if dist < self.capture_distance:
                        # Close enough, start capture sequence
                        self.state = "capturing";
                        self.target_humanoid.is_captured = True;
                        self.velocity = pygame.Vector2(0, 0) # Stop moving lander
                        # Humanoid starts moving towards capture point below lander
                        capture_point = self.position + pygame.Vector2(0, self.max_radius + 5) # Target pos below lander
                        humanoid_dir = (capture_point - self.target_humanoid.position)
                        if humanoid_dir.length() > 1:
                            self.target_humanoid.velocity = humanoid_dir.normalize() * 2.0 # Move humanoid
                        else: self.target_humanoid.velocity = pygame.Vector2(0,0)
                    else:
                        # Move towards target
                        if direction.length_squared() > 0: # Avoid normalizing zero vector
                            self.velocity = direction.normalize() * self.seek_speed
                        else: self.velocity = pygame.Vector2(0,0)
                else: # Target became unavailable (captured by another, fell)
                    self.find_target(humanoids_group);
                    self.state = "descending" if not self.target_humanoid else self.state # Re-descend if no new target
            else: # Target died or doesn't exist
                self.find_target(humanoids_group);
                self.state = "descending" if not self.target_humanoid else self.state

        elif self.state == "capturing":
             if self.target_humanoid and self.target_humanoid.alive() and self.target_humanoid.is_captured:
                 # Point slightly below the lander where humanoid should attach
                 capture_point = self.position + pygame.Vector2(0, self.max_radius + 5)
                 dist_sq = (capture_point - self.target_humanoid.position).length_squared()
                 # If humanoid is close enough to capture point, transition to ascending
                 if dist_sq < 10**2:
                      self.state = "ascending";
                      self.target_humanoid.velocity = pygame.Vector2(0,0) # Stop humanoid independent movement
                 else:
                      # Continue moving humanoid towards capture point
                      humanoid_dir = (capture_point - self.target_humanoid.position)
                      if humanoid_dir.length() > 1: self.target_humanoid.velocity = humanoid_dir.normalize() * 2.0
                      else: self.target_humanoid.velocity = pygame.Vector2(0,0)
             else: # Humanoid lost during capture (e.g., rescued by player)
                  self.state = "seeking"; self.find_target(humanoids_group)

        elif self.state == "ascending":
            # Move straight up
            self.velocity = pygame.Vector2(0, -self.seek_speed * 0.9) # Slightly slower ascent
            if self.target_humanoid and self.target_humanoid.alive() and self.target_humanoid.is_captured:
                # Humanoid position locked to lander
                self.target_humanoid.position = self.position + pygame.Vector2(0, self.max_radius + 5)
                self.target_humanoid.velocity = self.velocity # Humanoid moves with lander

                # --- Abduction Check ---
                if self.position.y < -self.max_radius * 2: # Lander fully off screen top
                    print(f"Lander escaped with humanoid at {self.position}!")
                    self.target_humanoid.kill() # Humanoid is lost (removed from groups)
                    self.target_humanoid = None
                    self.kill() # Lander leaves (removed from groups)
                    return # Stop further updates for this lander this frame
            else: # Humanoid lost during ascent (rescued)
                 self.target_humanoid = None; self.state = "seeking"; self.find_target(humanoids_group)

        # --- Firing Logic ---
        self.fire_timer -= 1
        # Only fire when descending or seeking (not capturing/ascending)
        if self.fire_timer <= 0 and self.state in ["descending", "seeking"]:
             self.fire(player_pos) # Pass player position for potential aiming later

        # Prevent lander going below terrain (unless ascending/capturing)
        if self.state not in ["ascending", "capturing"]:
            terrain_height = terrain_obj.get_height_at(self.position.x)
            if self.position.y + self.max_radius > terrain_height and self.velocity.y >= 0:
                self.position.y = terrain_height - self.max_radius # Sit on terrain
                if self.velocity.y > 0: self.velocity.y = 0 # Stop downward velocity

        super().update() # Applies velocity and world wrap for X

    def fire(self, player_pos):
        """ Fires an enemy laser downwards. """
        # Simple downward firing for now
        enemy_laser_velocity = pygame.Vector2(0, 5) # Speed 5 downwards
        # Fire from below the lander
        start_pos = self.position + pygame.Vector2(0, self.max_radius + 2)
        enemy_laser = EnemyLaser(start_pos, self.color, enemy_laser_velocity)
        all_sprites.add(enemy_laser); enemy_lasers.add(enemy_laser) # Add to groups
        # Reset fire timer, adjusted by speed multiplier
        speed_multiplier = self.seek_speed / 1.0 # Get current multiplier
        self.fire_timer = random.randint(int(self.fire_cooldown_base[0] / speed_multiplier), int(self.fire_cooldown_base[1] / speed_multiplier))


# --- EnemyLaser Class ---
class EnemyLaser(GameObject):
    """Represents a laser fired by an enemy."""
    def __init__(self, position, color, velocity):
        # Simple vertical line points
        points = [(0, -3), (0, 3)]; super().__init__(position, color, points)
        self.velocity = velocity; self.lifetime = 100 # Frames before disappearing
        # Angle based on velocity (usually vertical for enemy lasers)
        if velocity.length_squared() > 0: self.angle = math.atan2(velocity.y, velocity.x)
        else: self.angle = math.pi / 2 # Default vertical if no velocity

    def update(self, terrain_obj): # Add terrain_obj argument
        """ Update laser, check lifetime and terrain collision. """
        prev_pos = self.position.copy()
        super().update() # Applies velocity and world wrap
        self.lifetime -= 1

        if self.lifetime <= 0:
            self.kill()
            return

        # --- Terrain Collision Check ---
        current_terrain_y = terrain_obj.get_height_at(self.position.x)
        # Kill if laser position is below terrain
        if self.position.y > current_terrain_y:
            # print("Enemy laser hit terrain!") # DEBUG
            self.kill()


# --- Utility Functions ---
def draw_hud(surface):
    """ Draws the score, lives, and level display. """
    score_text = font.render(f"Score: {score}", True, WHITE);
    lives_text = font.render(f"Lives: {lives}", True, WHITE);
    level_text = font.render(f"Level: {level}", True, WHITE)
    surface.blit(score_text, (10, 10));
    surface.blit(lives_text, (10, 40));
    # Position level text relative to minimap
    surface.blit(level_text, (WIDTH - MINIMAP_WIDTH - 120, 10))

def draw_start_screen(surface):
    """ Draws the initial start screen. """
    title_text = large_font.render("VECTOR DEFENDER", True, GREEN);
    start_text = info_font.render("Press SPACE to Start", True, WHITE)
    controls_text1 = info_font.render("Arrows: Move Ship", True, WHITE);
    controls_text2 = info_font.render("Space: Fire", True, WHITE);
    controls_text3 = info_font.render("ESC: Quit", True, WHITE)
    # Center text elements
    title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 3));
    start_rect = start_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    controls1_rect = controls_text1.get_rect(center=(WIDTH // 2, HEIGHT * 0.65));
    controls2_rect = controls_text2.get_rect(center=(WIDTH // 2, HEIGHT * 0.72));
    controls3_rect = controls_text3.get_rect(center=(WIDTH // 2, HEIGHT * 0.79))
    surface.blit(title_text, title_rect); surface.blit(start_text, start_rect);
    surface.blit(controls_text1, controls1_rect); surface.blit(controls_text2, controls2_rect);
    surface.blit(controls_text3, controls3_rect)

def draw_game_over_screen(surface):
    """ Draws the game over screen. """
    game_over_text = large_font.render("GAME OVER", True, RED);
    score_text = info_font.render(f"Final Score: {score}", True, WHITE);
    restart_text = info_font.render("Press SPACE to Restart", True, WHITE)
    game_over_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 3));
    score_rect = score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2));
    restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT * 0.65))
    surface.blit(game_over_text, game_over_rect);
    surface.blit(score_text, score_rect);
    surface.blit(restart_text, restart_rect)

def draw_level_transition_screen(surface, current_level, saved_percent=None, bonus=None):
    """ Draws the screen shown between levels, including score bonus. """
    level_text = large_font.render(f"Level {current_level}", True, GREEN)
    level_rect = level_text.get_rect(center=(WIDTH // 2, HEIGHT // 3))
    surface.blit(level_text, level_rect)

    # Display bonus info if provided
    if saved_percent is not None and bonus is not None:
        percent_text = info_font.render(f"Humanoids Saved: {saved_percent:.0f}%", True, WHITE)
        bonus_text = info_font.render(f"Level Bonus: {bonus}", True, YELLOW)
        percent_rect = percent_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        bonus_rect = bonus_text.get_rect(center=(WIDTH // 2, HEIGHT * 0.6))
        surface.blit(percent_text, percent_rect)
        surface.blit(bonus_text, bonus_rect)

def draw_minimap(surface, player_obj, landers_group, humanoids_group):
    """ Draws the mini-map in the top-right corner. """
    # Create a separate surface for the minimap for transparency
    map_surf = pygame.Surface((MINIMAP_WIDTH, MINIMAP_HEIGHT), pygame.SRCALPHA)
    map_surf.fill(MINIMAP_BG) # Semi-transparent background
    pygame.draw.rect(map_surf, MINIMAP_BORDER, map_surf.get_rect(), 1) # Border

    # Draw humanoids (white dots)
    for humanoid in humanoids_group:
        map_x = int(humanoid.position.x * MINIMAP_SCALE_X)
        map_y = int(humanoid.position.y * MINIMAP_SCALE_Y)
        pygame.draw.circle(map_surf, WHITE, (map_x, map_y), 1)

    # Draw landers (red dots)
    for lander in landers_group:
        map_x = int(lander.position.x * MINIMAP_SCALE_X)
        map_y = int(lander.position.y * MINIMAP_SCALE_Y)
        pygame.draw.circle(map_surf, RED, (map_x, map_y), 1)

    # Draw player (green circle) if alive
    if not player_obj.is_destroyed:
        map_x = int(player_obj.position.x * MINIMAP_SCALE_X)
        map_y = int(player_obj.position.y * MINIMAP_SCALE_Y)
        pygame.draw.circle(map_surf, GREEN, (map_x, map_y), 2) # Player dot slightly larger

    # Blit the minimap surface onto the main screen
    surface.blit(map_surf, (MINIMAP_X, MINIMAP_Y))

# --- Game Initialization / Level Setup ---
def setup_level(current_level, is_new_game=False):
    """Sets up game objects for a specific level or new game."""
    # Make variables global that need to be accessed/modified
    global player, terrain, score, lives, camera_x, star_layer_far, star_layer_near
    global all_sprites, landers, humanoids, lasers, enemy_lasers, explosions
    global level_score, initial_humanoids_count # Add level score tracking

    if is_new_game:
        # Initialize game state for a brand new game
        score = 0; lives = 3; camera_x = 0
        terrain = Terrain(WORLD_WIDTH, HEIGHT)
        # Parallax star layers
        star_layer_far = ParallaxLayer(WORLD_WIDTH, 200, (100, 100, 100), (0, 1), 0.1)
        star_layer_near = ParallaxLayer(WORLD_WIDTH, 100, (200, 200, 200), (1, 2), 0.3)
        # Player setup
        player_start_x = WIDTH / 2; player_start_y = HEIGHT / 2
        player = Player((player_start_x, player_start_y))
        camera_x = player.position.x - WIDTH / 2 # Center camera initially
        # Sprite groups
        all_sprites = pygame.sprite.Group(); landers = pygame.sprite.Group();
        humanoids = pygame.sprite.Group(); lasers = pygame.sprite.Group();
        enemy_lasers = pygame.sprite.Group(); explosions = pygame.sprite.Group()
        all_sprites.add(player) # Add player to the main group
    else: # Setting up for the next level (not a new game)
        # Clear dynamic objects from previous level
        landers.empty(); lasers.empty(); enemy_lasers.empty(); explosions.empty()
        # Keep humanoids if any survived, they carry over implicitly unless abducted/killed
        # Remove specific object types explicitly from all_sprites
        for sprite in all_sprites.copy():
            if isinstance(sprite, (Lander, Laser, EnemyLaser, ExplosionParticle)):
                sprite.kill()
        # Humanoids that were killed/abducted are already removed by their own logic
        # Player respawn/positioning is handled after level transition delay

    level_score = 0 # Reset score accumulated *during* this level

    # --- Spawn objects for the current level ---
    num_landers_base = 10 # Increased base lander count
    num_landers_increase = 4 # Increased lander increment per level
    num_landers_for_level = num_landers_base + (current_level - 1) * num_landers_increase
    lander_speed_multiplier = 1.0 + (current_level - 1) * 0.08 # Landers get faster
    num_humanoids_base = 10 # Base number of humanoids
    # Add more humanoids every few levels
    num_humanoids_for_level = num_humanoids_base #+ (current_level // 2) # Keep constant for now?
    initial_humanoids_count = num_humanoids_for_level # Store initial count for bonus calc

    # Function to get a random spawn X across the whole world
    def get_random_spawn_x():
        return random.uniform(0, WORLD_WIDTH)

    # Spawn Humanoids (only if starting a new game or if all were lost? - current logic adds fixed number)
    # If not a new game, we should only add *new* humanoids, not replace existing ones.
    # Let's adjust: only spawn humanoids on level 1 / new game start.
    if is_new_game: # Only spawn initial set of humanoids at game start
        humanoids.empty() # Clear any leftovers if restarting
        for sprite in all_sprites.copy(): # Also remove from all_sprites
            if isinstance(sprite, Humanoid): sprite.kill()

        for _ in range(num_humanoids_for_level):
            humanoid_x = get_random_spawn_x()
            terrain_y = terrain.get_height_at(humanoid_x)
            humanoid_y = terrain_y - 10 # Place just above terrain
            # Ensure not spawning below screen bottom if terrain is high
            humanoid_y = min(humanoid_y, HEIGHT - 20)
            humanoid = Humanoid((humanoid_x, humanoid_y))
            humanoids.add(humanoid); all_sprites.add(humanoid)
    # Update initial count based on actual humanoids present when level *starts* playing
    # This should happen after transition, before gameplay loop starts for the level
    # Let's move this update to the transition logic end.

    # Spawn Landers for the level
    for _ in range(num_landers_for_level):
        lander_x = get_random_spawn_x()
        lander_y = random.uniform(30, 180) # Spawn near top of screen
        lander = Lander((lander_x, lander_y), lander_speed_multiplier)
        landers.add(lander); all_sprites.add(lander)

    print(f"--- Starting Level {current_level} ---");
    print(f"Landers: {num_landers_for_level}, Speed Multi: {lander_speed_multiplier:.2f}");
    # Humanoid count printed later when level actually starts

# --- Main Game Loop ---
setup_level(level, is_new_game=True) # Initial setup for Level 1
game_state = GAME_STATE_START
pygame.USEREVENT_GAME_OVER = pygame.USEREVENT + 1 # Custom event for game over delay
level_bonus_info = None # Store bonus info for transition screen

running = True
while running:
    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
            if event.key == pygame.K_SPACE:
                # Space starts game from start screen or restarts after game over
                if game_state == GAME_STATE_START:
                    level = 1; setup_level(level, is_new_game=True);
                    game_state = GAME_STATE_LEVEL_TRANSITION;
                    level_transition_timer = pygame.time.get_ticks();
                    level_bonus_info = None # No bonus on first level start
                elif game_state == GAME_STATE_GAME_OVER:
                    level = 1; setup_level(level, is_new_game=True);
                    game_state = GAME_STATE_LEVEL_TRANSITION;
                    level_transition_timer = pygame.time.get_ticks();
                    level_bonus_info = None
        # Handle the delayed game over event
        if event.type == pygame.USEREVENT_GAME_OVER:
             game_state = GAME_STATE_GAME_OVER

    current_time = pygame.time.get_ticks() # Get current time for timers

    # --- Game State Logic ---
    if game_state == GAME_STATE_PLAYING:
        keys = pygame.key.get_pressed()
        player.handle_input(keys)

        # --- Updates ---
        player.update()
        # Pass necessary info to updates (humanoids group, terrain, player pos)
        landers.update(humanoids, terrain, player.position)
        humanoids.update(terrain)
        # Pass terrain object to laser updates for collision check
        lasers.update(terrain)
        enemy_lasers.update(terrain)
        explosions.update() # Update active explosion particles

        # Update Camera Position based on player - TIGHT FOLLOW
        if not player.is_destroyed:
            # Camera center tries to match player's world X position
            camera_x = (player.position.x - WIDTH / 2)
            # Keep camera within world bounds (optional, prevents seeing "void" past world edges)
            # camera_x = max(0, min(camera_x, WORLD_WIDTH - WIDTH)) # Clamp camera


        # --- Collision Detection ---
        if not player.is_destroyed and not player.invulnerable: # Only check collisions if alive and vulnerable
            # Player vs Landers
            # spritecollideany returns the first lander colliding, or None
            collided_lander = pygame.sprite.spritecollideany(player, landers)
            if collided_lander:
                collided_lander.destroy(release_humanoid=True) # Ensure humanoid is released if carried
                player.crash() # Player also crashes

            # Player vs Enemy Lasers
            # spritecollide returns a list of lasers hit, True kills the lasers
            if pygame.sprite.spritecollide(player, enemy_lasers, True):
                 player.crash()

            # Player vs Terrain
            # Check point slightly below player center against terrain height
            player_bottom_y = player.position.y + player.max_radius * 0.8 # Approximate bottom
            terrain_y_at_player = terrain.get_height_at(player.position.x)
            if player_bottom_y >= terrain_y_at_player:
                 player.crash()

        # Player Lasers vs Landers
        # groupcollide checks for collisions between two groups
        # True kills the laser, False means lander isn't killed automatically (we handle it)
        lander_hits = pygame.sprite.groupcollide(lasers, landers, True, False)
        for laser, hit_landers_list in lander_hits.items():
            for lander_hit in hit_landers_list:
                level_score += 150 # Add score for this level
                # Lander's destroy method handles explosion and humanoid release
                lander_hit.destroy(release_humanoid=True)

        # Player Lasers vs Captured Humanoids (Rescue check)
        # Iterate through active lasers and check collision with humanoids
        for laser in lasers:
            # Check collision between this single laser and the humanoids group
            collided_humanoids = pygame.sprite.spritecollide(laser, humanoids, False) # Don't kill humanoid yet
            for humanoid in collided_humanoids:
                if humanoid.is_captured: # Only rescue captured ones
                    humanoid.is_captured = False; humanoid.is_falling = True # Release humanoid
                    level_score += 500 # Add rescue bonus to level score
                    laser.kill(); # Kill the laser that made the rescue
                    break # Laser can only rescue one per frame/hit

        # --- Check for Level Clear ---
        # Level clear only happens if player is alive and no Landers remain
        if game_state == GAME_STATE_PLAYING and not player.is_destroyed and not landers:
            print("Level Cleared!")

            # Calculate score bonus based on saved humanoids
            humanoids_saved = len(humanoids) # Count remaining humanoids
            # Use the count stored when the level *started*
            saved_percent = (humanoids_saved / initial_humanoids_count) * 100 if initial_humanoids_count > 0 else 100
            bonus_multiplier = saved_percent / 100.0
            # Bonus is based on score *earned during the level* multiplied by save %
            level_bonus = int(level_score * bonus_multiplier)
            score += level_bonus # Add bonus earned this level to total score
            level_bonus_info = (saved_percent, level_bonus) # Store info for display

            print(f"Humanoids Present at Level Start: {initial_humanoids_count}")
            print(f"Humanoids Saved: {humanoids_saved}/{initial_humanoids_count} ({saved_percent:.0f}%)")
            print(f"Level Score: {level_score}, Multiplier: {bonus_multiplier:.2f}, Bonus Added: {level_bonus}, Total Score: {score}")

            # Advance level and transition
            level += 1
            game_state = GAME_STATE_LEVEL_TRANSITION
            level_transition_timer = current_time

        # --- Drawing ---
        screen.fill(BLACK) # Clear screen
        # Draw parallax stars first (background)
        star_layer_far.draw(screen, camera_x)
        star_layer_near.draw(screen, camera_x)
        # Draw terrain (covers bottom part, including stars below it)
        terrain.draw(screen, camera_x)
        # Draw all sprites using their individual draw methods which handle wrap & camera
        for sprite in all_sprites:
            # Player draw method handles its own visibility flash during invulnerability
            sprite.draw(screen, camera_x)
        # Draw HUD and Minimap last (on top)
        draw_hud(screen)
        draw_minimap(screen, player, landers, humanoids)

    # --- Other Game States ---
    elif game_state == GAME_STATE_PLAYER_DIED:
        # Keep updating explosions during death sequence
        explosions.update()
        # Update camera slightly? Or keep it fixed? Let's keep it fixed for now.
        # camera_x = (player.position.x - WIDTH / 2) # Last position before death

        # Check respawn timer
        if current_time - respawn_timer > RESPAWN_DELAY:
            # Respawn player near the center of the current camera view
            respawn_x = (camera_x + WIDTH / 2) % WORLD_WIDTH
            respawn_y = HEIGHT * 0.4 # Respawn higher up
            player.respawn((respawn_x, respawn_y))
            game_state = GAME_STATE_PLAYING # Return to playing state

        # Drawing during death state (show background, terrain, explosions)
        screen.fill(BLACK)
        star_layer_far.draw(screen, camera_x)
        star_layer_near.draw(screen, camera_x)
        terrain.draw(screen, camera_x)
        # Draw remaining objects except the player
        for sprite in all_sprites:
             if sprite != player: # Don't draw destroyed player
                 sprite.draw(screen, camera_x)
        # Draw explosions on top
        for explosion in explosions:
             explosion.draw(screen, camera_x)
        draw_hud(screen) # Still show HUD

    elif game_state == GAME_STATE_LEVEL_TRANSITION:
        screen.fill(BLACK)
        # Display level transition screen with bonus info if available
        if level_bonus_info:
            draw_level_transition_screen(screen, level, level_bonus_info[0], level_bonus_info[1])
        else:
            draw_level_transition_screen(screen, level) # Show only level number if starting game

        # After delay, setup next level and respawn player
        if current_time - level_transition_timer > LEVEL_TRANSITION_DURATION:
            setup_level(level, is_new_game=False) # Setup enemies etc. for new level
            # Store actual humanoid count now level objects are set up
            initial_humanoids_count = len(humanoids)
            print(f"Humanoids Present at Start of Level {level}: {initial_humanoids_count}")

            # Respawn player in the middle of the screen view for the new level
            respawn_x = (camera_x + WIDTH / 2) % WORLD_WIDTH # Use current camera pos
            respawn_y = HEIGHT * 0.4
            player.respawn((respawn_x, respawn_y)) # Player appears, invulnerable
            level_bonus_info = None # Clear bonus info after displaying it
            game_state = GAME_STATE_PLAYING # Start playing the new level

    elif game_state == GAME_STATE_START:
        screen.fill(BLACK); draw_start_screen(screen)

    elif game_state == GAME_STATE_GAME_OVER:
        # Update explosions during game over screen
        explosions.update()
        # Draw background elements and explosions
        screen.fill(BLACK)
        star_layer_far.draw(screen, camera_x);
        star_layer_near.draw(screen, camera_x);
        terrain.draw(screen, camera_x)
        for explosion in explosions: explosion.draw(screen, camera_x)
        # Draw the centered game over text
        draw_game_over_screen(screen)

    # Update the full display Surface to the screen
    pygame.display.flip()
    # Limit frame rate
    clock.tick(FPS)

# --- End of Game ---
pygame.quit()
sys.exit()