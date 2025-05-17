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

        screen_points = []
        for wx, wy in world_points:
            sx = wx - camera_offset_x
            sy = wy
            screen_points.append((int(sx), int(sy)))

        self._draw_polygon_if_visible(surface, screen_points)

        # Determine min/max world coordinates for wrap checking
        # This needs to handle the case where world_points might be empty if base_points is empty
        if world_points:
            world_x_coords = [p[0] for p in world_points]
            min_world_x = min(world_x_coords)
            max_world_x = max(world_x_coords)
        else: # Fallback for objects with no explicit points (e.g. some particles before refactor)
            min_world_x = self.position.x - self.max_radius
            max_world_x = self.position.x + self.max_radius


        # Condition 1: Object is near the LEFT edge of the world, and camera is viewing the RIGHT end of the world
        if min_world_x < self.max_radius * 2 and camera_offset_x > WORLD_WIDTH - WIDTH:
            wrapped_screen_points = [(p[0] + WORLD_WIDTH, p[1]) for p in screen_points]
            self._draw_polygon_if_visible(surface, wrapped_screen_points)

        # Condition 2: Object is near the RIGHT edge of the world, and camera is viewing the LEFT start of the world
        elif max_world_x > WORLD_WIDTH - self.max_radius * 2 and camera_offset_x < WIDTH:
            wrapped_screen_points = [(p[0] - WORLD_WIDTH, p[1]) for p in screen_points]
            self._draw_polygon_if_visible(surface, wrapped_screen_points)


    def _draw_polygon_if_visible(self, surface, screen_points):
        """ Helper function to draw polygon only if its bounding box overlaps the screen. """
        line_width = 2 if isinstance(self, Laser) else 1

        if not screen_points or len(screen_points) < 2: return

        min_x = min(p[0] for p in screen_points)
        max_x = max(p[0] for p in screen_points)
        min_y = min(p[1] for p in screen_points)
        max_y = max(p[1] for p in screen_points)

        if max_x > 0 and min_x < WIDTH and max_y > 0 and min_y < HEIGHT:
            try:
                if len(screen_points) >= 3:
                    pygame.draw.polygon(surface, self.color, screen_points, line_width)
                elif len(screen_points) == 2:
                    pygame.draw.line(surface, self.color, screen_points[0], screen_points[1], line_width)
            except ValueError:
                pass # Silently ignore ValueErrors

    def update(self):
        """Updates world position, handles world wrap, and updates rect."""
        self.position += self.velocity
        self.position.x %= WORLD_WIDTH
        self.rect.center = (int(self.position.x), int(self.position.y))

    def destroy(self, release_humanoid=True):
        play_sound(explosion_sound)
        num_particles = random.randint(15, 25)
        for _ in range(num_particles):
            particle = ExplosionParticle(self.position.copy(), self.color)
            all_sprites.add(particle)
            explosions.add(particle)

        if release_humanoid and isinstance(self, Lander) and self.target_humanoid and self.target_humanoid.alive() and self.target_humanoid.is_captured:
            self.target_humanoid.is_captured = False
            self.target_humanoid.is_falling = True
            self.target_humanoid = None

        self.kill()


# --- Player Class ---
class Player(GameObject):
    """Represents the player ship."""
    def __init__(self, position):
        self.points_right = [ (15, 0), (-12, -8), (-7, 0), (-12, 8) ]
        self.points_left = [(-p[0], p[1]) for p in self.points_right]
        super().__init__(position, GREEN, self.points_right)
        self.facing_direction = 1
        self.base_points = self.points_right
        self.acceleration = 0.28; self.friction = 0.95; self.max_speed = 7
        self.fire_cooldown = 180
        self.can_fire = True; self.fire_cooldown_timer = 0
        self.accel = pygame.Vector2(0, 0)
        self.is_destroyed = False
        self.invulnerable = False
        self.invulnerable_timer = 0
        self.INVULNERABLE_DURATION = 1500
        self.visible = True

    def handle_input(self, keys):
        if self.is_destroyed: return
        self.accel.x = 0; self.accel.y = 0
        new_facing_direction = self.facing_direction
        if keys[pygame.K_LEFT]: self.accel.x = -self.acceleration; new_facing_direction = -1
        if keys[pygame.K_RIGHT]: self.accel.x = self.acceleration; new_facing_direction = 1
        if keys[pygame.K_UP]: self.accel.y = -self.acceleration
        if keys[pygame.K_DOWN]: self.accel.y = self.acceleration

        if new_facing_direction != self.facing_direction:
            self.facing_direction = new_facing_direction
            self.base_points = self.points_left if self.facing_direction == -1 else self.points_right
            self.max_radius = max(math.hypot(p[0], p[1]) for p in self.base_points) if self.base_points else 1

        if keys[pygame.K_SPACE] and self.can_fire:
             self.fire(); self.can_fire = False; self.fire_cooldown_timer = pygame.time.get_ticks()

    def fire(self):
        play_sound(pew_sound)
        offset_magnitude = 18
        laser_start_offset = pygame.Vector2(offset_magnitude * self.facing_direction, 0)
        laser_position = (self.position + laser_start_offset)
        laser_position.x %= WORLD_WIDTH

        laser_speed = 14
        base_laser_velocity = pygame.Vector2(laser_speed * self.facing_direction, 0)
        laser_velocity = base_laser_velocity + self.velocity * 0.5
        laser = Laser(laser_position, LASER_COLOR, laser_velocity)
        all_sprites.add(laser); lasers.add(laser)

    def update(self):
        current_time = pygame.time.get_ticks()
        if self.is_destroyed: return

        if self.invulnerable:
            if current_time - self.invulnerable_timer > self.INVULNERABLE_DURATION:
                self.invulnerable = False; self.visible = True
            else:
                self.visible = (current_time // 100) % 2 == 0

        self.velocity += self.accel; self.velocity *= self.friction
        if self.velocity.length_squared() < 0.01: self.velocity.xy = (0, 0)
        if self.velocity.length_squared() > self.max_speed**2: self.velocity.scale_to_length(self.max_speed)

        super().update()

        if self.position.y > HEIGHT - self.max_radius: self.position.y = HEIGHT - self.max_radius; self.velocity.y = 0
        elif self.position.y < self.max_radius: self.position.y = self.max_radius; self.velocity.y = 0

        if not self.can_fire and current_time - self.fire_cooldown_timer > self.fire_cooldown:
             self.can_fire = True

    def crash(self):
        global lives, game_state, respawn_timer
        if not self.is_destroyed and not self.invulnerable:
            self.is_destroyed = True; lives -= 1
            self.destroy(release_humanoid=False)
            if lives > 0:
                game_state = GAME_STATE_PLAYER_DIED; respawn_timer = pygame.time.get_ticks()
            else:
                pygame.time.set_timer(pygame.USEREVENT_GAME_OVER, 1000, 1)

    def respawn(self, respawn_pos):
        self.position = pygame.Vector2(respawn_pos); self.velocity = pygame.Vector2(0, 0)
        self.is_destroyed = False; self.can_fire = True; self.facing_direction = 1
        self.base_points = self.points_right
        self.invulnerable = True
        self.invulnerable_timer = pygame.time.get_ticks(); self.visible = True
        if self not in all_sprites: all_sprites.add(self)

    def draw(self, surface, camera_offset_x):
        if self.visible:
            super().draw(surface, camera_offset_x)


# --- Explosion Particle Class ---
class ExplosionParticle(GameObject):
    def __init__(self, position, base_color):
        super().__init__(position, base_color, points=None) # No base_points for circle
        speed = random.uniform(2, 6); angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifetime = random.randint(20, 40)
        self.start_radius = random.randint(2, 4)
        self.radius = self.start_radius
        self.color = random.choice([RED, ORANGE, YELLOW, WHITE])
        self.max_radius = self.start_radius # For consistency if GameObject.draw was used

    def update(self):
        super().update() # Applies velocity and world wrap
        self.lifetime -= 1
        self.radius = self.start_radius * (self.lifetime / 40.0) # Shrink
        if self.lifetime <= 0 or self.radius < 1:
            self.kill()

    def draw(self, surface, camera_offset_x):
        # Custom draw for circle, handles simple wrap for single position
        # For particles, drawing one instance at the "closest" wrapped position is often enough
        dx = self.position.x - camera_offset_x
        if dx > WORLD_WIDTH / 2: dx -= WORLD_WIDTH
        elif dx < -WORLD_WIDTH / 2: dx += WORLD_WIDTH
        screen_x = int(dx)
        screen_y = int(self.position.y)

        if self.radius >= 1:
            # Basic visibility check for the particle's current screen position
            if -self.radius < screen_x < WIDTH + self.radius and \
               -self.radius < screen_y < HEIGHT + self.radius:
                pygame.draw.circle(surface, self.color, (screen_x, screen_y), int(self.radius))


# --- Laser Class ---
class Laser(GameObject):
    def __init__(self, position, color, velocity):
        points = [(0, 0), (12, 0)]
        super().__init__(position, color, points)
        self.velocity = velocity
        self.lifetime = FPS # 1-second lifetime

        if velocity.length_squared() > 0: self.angle = math.atan2(velocity.y, velocity.x)
        else: self.angle = 0

    def update(self, terrain_obj, camera_offset_x):
        super().update()

        if hasattr(self, 'rect'):
            if self.rect.right < camera_offset_x or \
               self.rect.left > camera_offset_x + WIDTH:
                self.kill()
                return

        self.lifetime -= 1
        if self.lifetime <= 0:
            self.kill()
            return

        current_terrain_y = terrain_obj.get_height_at(self.position.x)
        if self.position.y > current_terrain_y:
            self.kill()


# --- Terrain Class ---
class Terrain:
    def __init__(self, world_width, screen_height, segment_length=25):
        self.world_width = world_width; self.screen_height = screen_height
        self.segment_length = segment_length
        self.points = self._generate_terrain() # List of (world_x, world_y) tuples

    def _generate_terrain(self):
        # (Generation logic remains the same as in your provided code)
        points = []; x = 0; y = self.screen_height * 0.80
        min_y = self.screen_height * 0.50; max_y = self.screen_height - 60
        slope = 0
        while x < self.world_width:
            points.append((x, int(y)))
            change_type = random.random()
            if change_type < 0.05:
                max_deviation = 100; dy = random.randint(-max_deviation, max_deviation); slope = 0
            elif change_type < 0.15:
                 dy = 0; slope = 0
            elif change_type < 0.5:
                max_deviation = 40; slope_change = random.randint(-max_deviation // 2, max_deviation // 2)
                if slope > 0: slope = max(-max_deviation, slope - slope_change - 5)
                elif slope < 0: slope = min(max_deviation, slope + slope_change + 5)
                else: slope += slope_change
                dy = slope
            else:
                max_deviation = 40; dy = slope + random.randint(-max_deviation // 4, max_deviation // 4)
            y += dy;
            y = max(min_y, min(y, max_y));
            x += self.segment_length
        points.append((self.world_width, points[0][1])) # Ensure wrap for interpolation
        return points

    def get_height_at(self, world_x):
        # (get_height_at logic remains the same)
        world_x %= self.world_width
        index = int(world_x // self.segment_length)
        index = max(0, min(index, len(self.points) - 2)) # Max index is len-2 for p1, p2
        p1 = self.points[index]; p2 = self.points[index + 1]
        if p2[0] == p1[0]: return p1[1] # Vertical segment
        segment_width = p2[0] - p1[0]
        if segment_width <= 0: segment_width = self.segment_length # Fallback
        local_x = world_x - p1[0]
        t = local_x / segment_width
        t = max(0.0, min(1.0, t)) # Clamp t for interpolation
        height = p1[1] + t * (p2[1] - p1[1])
        return int(height)

    def _draw_terrain_pass(self, surface, point_list_for_pass):
        """Helper to draw lines and fill for a given list of screen points."""
        if not point_list_for_pass or len(point_list_for_pass) < 2:
            return

        # Filter points to be roughly on screen or near it to avoid drawing far-off segments.
        # The range [-WIDTH, WIDTH*2] allows for points starting off-screen left
        # and extending beyond off-screen right.
        drawable_points = [p for p in point_list_for_pass if -WIDTH <= p[0] <= WIDTH * 2]
        
        # Sort by x-coordinate, crucial for pygame.draw.lines and polygon fill
        drawable_points.sort(key=lambda pt: pt[0])

        if not drawable_points or len(drawable_points) < 2:
            return

        # Draw the filled polygon (black area below terrain)
        # Bottom corners extend from the first and last *drawable* point's x-coordinate
        bottom_left = (drawable_points[0][0], HEIGHT)
        bottom_right = (drawable_points[-1][0], HEIGHT)
        
        # Construct the polygon points: bottom-left, terrain points, bottom-right
        final_fill_points = [bottom_left] + drawable_points + [bottom_right]
        
        if len(final_fill_points) >= 3:
            try:
                pygame.draw.polygon(surface, BLACK, final_fill_points, 0) # 0 width = fill
            except ValueError:
                pass # Ignore errors from bad coordinate lists

        # Draw the terrain surface line (blue outline)
        try:
            # Use draw.lines for connected segments, False means not closed loop
            pygame.draw.lines(surface, BLUE, False, drawable_points, 2) # Thickness 2
        except ValueError:
            pass # Silently ignore

    def draw(self, surface, camera_offset_x):
        """ Draws the terrain line and fills below, handling wrap correctly using multiple passes. """
        # 1. Generate base screen points: all terrain points transformed relative to camera_offset_x
        base_screen_points = []
        for world_x_coord, world_y_coord in self.points: # self.points are world coordinates
            screen_x_coord = world_x_coord - camera_offset_x
            base_screen_points.append((int(screen_x_coord), int(world_y_coord)))

        # 2. Draw the main pass
        self._draw_terrain_pass(surface, base_screen_points)

        # 3. Draw wrapped pass (for terrain appearing from the "left" if camera is far right)
        # These points are shifted as if the world repeated to the right
        points_wrapped_plus_world = [(sx + WORLD_WIDTH, sy) for sx, sy in base_screen_points]
        self._draw_terrain_pass(surface, points_wrapped_plus_world)
        
        # 4. Draw wrapped pass (for terrain appearing from the "right" if camera is far left)
        # These points are shifted as if the world repeated to the left
        points_wrapped_minus_world = [(sx - WORLD_WIDTH, sy) for sx, sy in base_screen_points]
        self._draw_terrain_pass(surface, points_wrapped_minus_world)


# --- ParallaxLayer Class ---
class ParallaxLayer:
    # (ParallaxLayer class remains the same as in your provided code)
    def __init__(self, world_width, num_elements, color, size_range, scroll_factor):
        self.elements = []; self.color = color; self.scroll_factor = scroll_factor
        for _ in range(num_elements):
            x = random.uniform(0, world_width); y = random.uniform(0, HEIGHT)
            size = random.randint(size_range[0], size_range[1])
            self.elements.append({'x': x, 'y': y, 'size': size})

    def draw(self, surface, camera_offset_x):
        for element in self.elements:
            world_x, world_y, size = element['x'], element['y'], element['size']
            parallax_cam_x = camera_offset_x * self.scroll_factor
            
            # Calculate screen_x by finding the closest wrapped instance
            direct_sx = world_x - parallax_cam_x
            options = [direct_sx, direct_sx + WORLD_WIDTH, direct_sx - WORLD_WIDTH]
            
            screen_x_final = min(options, key=lambda val: abs(val - WIDTH / 2))


            if -size <= screen_x_final <= WIDTH + size and \
               -size <= world_y <= HEIGHT + size : # Also check Y for sanity
                if size <= 1:
                    try: surface.set_at((int(screen_x_final), int(world_y)), self.color)
                    except IndexError: pass
                else:
                    pygame.draw.circle(surface, self.color, (int(screen_x_final), int(world_y)), size)


# --- Humanoid Class ---
class Humanoid(GameObject):
    # (Humanoid class remains the same)
    def __init__(self, position):
        points = [ (0, -8), (0, 0), (-4, 5), (0, 0), (4, 5), (0, 0), (-5, -4), (0, 0), (5, -4) ]
        super().__init__(position, WHITE, points)
        self.is_captured = False; self.is_falling = False; self.fall_speed = 2.5;
        self.capture_target_y = -30

    def update(self, terrain_obj):
        if self.is_captured:
            if self.position.y < self.capture_target_y:
                self.kill()
        elif self.is_falling:
            self.velocity = pygame.Vector2(0, self.fall_speed)
            terrain_y = terrain_obj.get_height_at(self.position.x)
            if self.position.y >= terrain_y - self.max_radius:
                self.position.y = terrain_y - self.max_radius; self.is_falling = False; self.velocity = pygame.Vector2(0, 0)
        else: # On ground
            self.velocity = pygame.Vector2(0, 0)
            terrain_y = terrain_obj.get_height_at(self.position.x)
            if abs(self.position.y - (terrain_y - self.max_radius)) > 2: # Tolerance for float issues
                 self.position.y = terrain_y - self.max_radius
        super().update()

# --- Lander Class ---
class Lander(GameObject):
    # (Lander class remains the same)
    def __init__(self, position, speed_multiplier=1.0):
        points = [ (-8, 8), (-10, 0), (-8, -8), (8, -8), (10, 0), (8, 8), (5, 8), (0, 12), (-5, 8) ]
        super().__init__(position, RED, points)
        self.state = "descending"; self.target_humanoid = None
        self.seek_speed = 1.0 * speed_multiplier
        self.descent_speed = 0.5 * speed_multiplier
        self.capture_distance = 20
        self.fire_timer = random.randint(int(90 / speed_multiplier), int(240 / speed_multiplier))
        self.fire_cooldown_base = (90, 240);
        self.horizontal_drift_speed = 0.4 * speed_multiplier
        self.angle = random.uniform(-0.1, 0.1)

    def find_target(self, humanoids_group):
        self.target_humanoid = None; min_distance_sq = float('inf')
        for humanoid in humanoids_group:
            if humanoid.alive() and not humanoid.is_captured and not humanoid.is_falling:
                dx = abs(self.position.x - humanoid.position.x)
                wrapped_dx = min(dx, WORLD_WIDTH - dx)
                dy = abs(self.position.y - humanoid.position.y)
                distance_sq = wrapped_dx**2 + dy**2
                if distance_sq < min_distance_sq:
                     min_distance_sq = distance_sq; self.target_humanoid = humanoid

    def update(self, humanoids_group, terrain_obj, player_pos):
        if self.state == "descending":
            self.velocity.y = self.descent_speed
            if abs(self.velocity.x) < self.horizontal_drift_speed * 0.5 and random.random() < 0.02:
                 self.velocity.x = random.uniform(-self.horizontal_drift_speed, self.horizontal_drift_speed)
            else: self.velocity.x *= 0.95
            terrain_height = terrain_obj.get_height_at(self.position.x)
            if self.position.y > terrain_height - HEIGHT * 0.6:
                self.find_target(humanoids_group)
                if self.target_humanoid:
                     self.state = "seeking"; self.velocity.x = 0
                elif self.position.y > terrain_height - self.max_radius * 3:
                     self.velocity.y = 0
                     if abs(self.velocity.x) < 0.1:
                         self.velocity.x = random.uniform(-self.horizontal_drift_speed*0.5, self.horizontal_drift_speed*0.5)
        elif self.state == "seeking":
            if self.target_humanoid and self.target_humanoid.alive():
                if not self.target_humanoid.is_captured and not self.target_humanoid.is_falling:
                    dx = self.target_humanoid.position.x - self.position.x
                    if dx > WORLD_WIDTH / 2: dx -= WORLD_WIDTH
                    elif dx < -WORLD_WIDTH / 2: dx += WORLD_WIDTH
                    dy = self.target_humanoid.position.y - self.position.y
                    direction = pygame.Vector2(dx, dy)
                    dist = direction.length()
                    if dist < self.capture_distance:
                        self.state = "capturing";
                        self.target_humanoid.is_captured = True;
                        self.velocity = pygame.Vector2(0, 0)
                        capture_point = self.position + pygame.Vector2(0, self.max_radius + 5)
                        humanoid_dir = (capture_point - self.target_humanoid.position)
                        if humanoid_dir.length() > 1: self.target_humanoid.velocity = humanoid_dir.normalize() * 2.0
                        else: self.target_humanoid.velocity = pygame.Vector2(0,0)
                    else:
                        if direction.length_squared() > 0: self.velocity = direction.normalize() * self.seek_speed
                        else: self.velocity = pygame.Vector2(0,0)
                else:
                    self.find_target(humanoids_group);
                    self.state = "descending" if not self.target_humanoid else self.state
            else:
                self.find_target(humanoids_group);
                self.state = "descending" if not self.target_humanoid else self.state
        elif self.state == "capturing":
             if self.target_humanoid and self.target_humanoid.alive() and self.target_humanoid.is_captured:
                 capture_point = self.position + pygame.Vector2(0, self.max_radius + 5)
                 dist_sq = (capture_point - self.target_humanoid.position).length_squared()
                 if dist_sq < 10**2: # Humanoid close enough to be "attached"
                      self.state = "ascending";
                      self.target_humanoid.velocity = pygame.Vector2(0,0) # Stop humanoid independent movement
                 else: # Continue moving humanoid towards capture point
                      humanoid_dir = (capture_point - self.target_humanoid.position)
                      if humanoid_dir.length() > 1: self.target_humanoid.velocity = humanoid_dir.normalize() * 2.0
                      else: self.target_humanoid.velocity = pygame.Vector2(0,0)
             else: # Humanoid lost or rescued during capture process
                  self.state = "seeking"; self.find_target(humanoids_group)
        elif self.state == "ascending":
            self.velocity = pygame.Vector2(0, -self.seek_speed * 0.9) # Ascend
            if self.target_humanoid and self.target_humanoid.alive() and self.target_humanoid.is_captured:
                self.target_humanoid.position = self.position + pygame.Vector2(0, self.max_radius + 5) # Humanoid moves with lander
                self.target_humanoid.velocity = self.velocity # Match lander's velocity
                if self.position.y < -self.max_radius * 2: # Lander fully off screen top
                    self.target_humanoid.kill() # Humanoid is abducted
                    self.target_humanoid = None
                    self.kill() # Lander leaves
                    return # Stop further updates for this lander
            else: # Humanoid lost or rescued during ascent
                 self.target_humanoid = None; self.state = "seeking"; self.find_target(humanoids_group)

        self.fire_timer -= 1
        if self.fire_timer <= 0 and self.state in ["descending", "seeking"]:
             self.fire(player_pos)

        if self.state not in ["ascending", "capturing"]: # Collision with terrain
            terrain_height = terrain_obj.get_height_at(self.position.x)
            if self.position.y + self.max_radius > terrain_height and self.velocity.y >= 0 : # If moving down or on ground
                self.position.y = terrain_height - self.max_radius # Sit on terrain
                if self.velocity.y > 0: self.velocity.y = 0 # Stop downward velocity
        super().update()

    def fire(self, player_pos):
        enemy_laser_velocity = pygame.Vector2(0, 5)
        start_pos = self.position + pygame.Vector2(0, self.max_radius + 2)
        enemy_laser = EnemyLaser(start_pos, self.color, enemy_laser_velocity)
        all_sprites.add(enemy_laser); enemy_lasers.add(enemy_laser)
        speed_multiplier = self.seek_speed / 1.0 # Base seek speed is 1.0
        self.fire_timer = random.randint(int(self.fire_cooldown_base[0] / max(0.1, speed_multiplier)), \
                                         int(self.fire_cooldown_base[1] / max(0.1, speed_multiplier)))


# --- EnemyLaser Class ---
class EnemyLaser(GameObject):
    # (EnemyLaser class remains mostly the same, uses updated Laser lifetime and update)
    def __init__(self, position, color, velocity):
        points = [(0, -3), (0, 3)]
        super().__init__(position, color, points)
        self.velocity = velocity
        self.lifetime = FPS # 1-second lifetime
        if velocity.length_squared() > 0: self.angle = math.atan2(velocity.y, velocity.x)
        else: self.angle = math.pi / 2

    def update(self, terrain_obj, camera_offset_x):
        super().update()

        if hasattr(self, 'rect'):
            if self.rect.right < camera_offset_x or \
               self.rect.left > camera_offset_x + WIDTH:
                self.kill()
                return

        self.lifetime -= 1
        if self.lifetime <= 0:
            self.kill()
            return

        current_terrain_y = terrain_obj.get_height_at(self.position.x)
        if self.position.y > current_terrain_y:
            self.kill()


# --- Utility Functions ---
# (draw_hud, draw_start_screen, draw_game_over_screen, draw_level_transition_screen, draw_minimap remain the same)
def draw_hud(surface):
    score_text = font.render(f"Score: {score}", True, WHITE);
    lives_text = font.render(f"Lives: {lives}", True, WHITE);
    level_text = font.render(f"Level: {level}", True, WHITE)
    surface.blit(score_text, (10, 10));
    surface.blit(lives_text, (10, 40));
    surface.blit(level_text, (WIDTH - MINIMAP_WIDTH - 120, 10))

def draw_start_screen(surface):
    title_text = large_font.render("VECTOR DEFENDER", True, GREEN);
    start_text = info_font.render("Press SPACE to Start", True, WHITE)
    controls_text1 = info_font.render("Arrows: Move Ship", True, WHITE);
    controls_text2 = info_font.render("Space: Fire", True, WHITE);
    controls_text3 = info_font.render("ESC: Quit", True, WHITE)
    title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 3));
    start_rect = start_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    controls1_rect = controls_text1.get_rect(center=(WIDTH // 2, HEIGHT * 0.65));
    controls2_rect = controls_text2.get_rect(center=(WIDTH // 2, HEIGHT * 0.72));
    controls3_rect = controls_text3.get_rect(center=(WIDTH // 2, HEIGHT * 0.79))
    surface.blit(title_text, title_rect); surface.blit(start_text, start_rect);
    surface.blit(controls_text1, controls1_rect); surface.blit(controls_text2, controls2_rect);
    surface.blit(controls_text3, controls3_rect)

def draw_game_over_screen(surface):
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
    level_text = large_font.render(f"Level {current_level}", True, GREEN)
    level_rect = level_text.get_rect(center=(WIDTH // 2, HEIGHT // 3))
    surface.blit(level_text, level_rect)
    if saved_percent is not None and bonus is not None:
        percent_text = info_font.render(f"Humanoids Saved: {saved_percent:.0f}%", True, WHITE)
        bonus_text = info_font.render(f"Level Bonus: {bonus}", True, YELLOW)
        percent_rect = percent_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        bonus_rect = bonus_text.get_rect(center=(WIDTH // 2, HEIGHT * 0.6))
        surface.blit(percent_text, percent_rect)
        surface.blit(bonus_text, bonus_rect)

def draw_minimap(surface, player_obj, landers_group, humanoids_group):
    map_surf = pygame.Surface((MINIMAP_WIDTH, MINIMAP_HEIGHT), pygame.SRCALPHA)
    map_surf.fill(MINIMAP_BG)
    pygame.draw.rect(map_surf, MINIMAP_BORDER, map_surf.get_rect(), 1)
    for humanoid_obj in humanoids_group: # Renamed to avoid conflict
        map_x = int(humanoid_obj.position.x * MINIMAP_SCALE_X)
        map_y = int(humanoid_obj.position.y * MINIMAP_SCALE_Y)
        pygame.draw.circle(map_surf, WHITE, (map_x, map_y), 1)
    for lander_obj in landers_group: # Renamed
        map_x = int(lander_obj.position.x * MINIMAP_SCALE_X)
        map_y = int(lander_obj.position.y * MINIMAP_SCALE_Y)
        pygame.draw.circle(map_surf, RED, (map_x, map_y), 1)
    if not player_obj.is_destroyed:
        map_x = int(player_obj.position.x * MINIMAP_SCALE_X)
        map_y = int(player_obj.position.y * MINIMAP_SCALE_Y)
        pygame.draw.circle(map_surf, GREEN, (map_x, map_y), 2)
    surface.blit(map_surf, (MINIMAP_X, MINIMAP_Y))

# --- Game Initialization / Level Setup ---
def setup_level(current_level, is_new_game=False):
    global player, terrain, score, lives, camera_x, star_layer_far, star_layer_near
    global all_sprites, landers, humanoids, lasers, enemy_lasers, explosions
    global level_score # initial_humanoids_count is set after this function

    if is_new_game:
        # Initialize game state for a brand new game
        score = 0; lives = 3; camera_x = 0 # Reset these crucial game state vars
        terrain = Terrain(WORLD_WIDTH, HEIGHT)
        star_layer_far = ParallaxLayer(WORLD_WIDTH, 200, (100, 100, 100), (0, 1), 0.1)
        star_layer_near = ParallaxLayer(WORLD_WIDTH, 100, (200, 200, 200), (1, 2), 0.3)
        
        player_start_x = WIDTH / 2; player_start_y = HEIGHT / 2
        player = Player((player_start_x, player_start_y)) # Create new player instance
        camera_x = player.position.x - WIDTH / 2 # Center camera on new player
        
        # Initialize sprite groups for a new game
        all_sprites = pygame.sprite.Group(); landers = pygame.sprite.Group();
        humanoids = pygame.sprite.Group(); lasers = pygame.sprite.Group();
        enemy_lasers = pygame.sprite.Group(); explosions = pygame.sprite.Group()
        all_sprites.add(player) # Add the new player to all_sprites
    else: # Setting up for the next level (not a brand new game)
        # Clear dynamic objects from previous level
        # Player, terrain, stars, score, lives persist across levels (unless game over)
        landers.empty(); lasers.empty(); enemy_lasers.empty(); explosions.empty()
        # Remove specific object types explicitly from all_sprites
        for sprite_obj in all_sprites.copy(): # Use sprite_obj to avoid conflict
            if isinstance(sprite_obj, (Lander, Laser, EnemyLaser, ExplosionParticle)):
                sprite_obj.kill()
        # Humanoids are handled next, to be fully repopulated.

    level_score = 0 # Reset score accumulated *during* this level

    # --- Humanoid Repopulation for ALL levels ---
    humanoids.empty() # Clear any humanoids from previous level
    for sprite_obj in all_sprites.copy(): # Ensure all_sprites is also cleared of old humanoids
        if isinstance(sprite_obj, Humanoid):
            sprite_obj.kill()

    num_humanoids_base = 10
    num_humanoids_for_level = num_humanoids_base # Can be adjusted based on level later
    # initial_humanoids_count will be set based on len(humanoids) after this function,
    # typically in the level transition logic.

    def get_random_spawn_x(): # Helper for random x position in the world
        return random.uniform(0, WORLD_WIDTH)

    for _ in range(num_humanoids_for_level):
        humanoid_x = get_random_spawn_x()
        terrain_y_at_x = terrain.get_height_at(humanoid_x) # Renamed for clarity
        humanoid_y = terrain_y_at_x - 10 # Place just above terrain
        humanoid_y = min(humanoid_y, HEIGHT - 20) # Ensure not spawning below screen bottom
        new_humanoid_obj = Humanoid((humanoid_x, humanoid_y)) # Renamed
        humanoids.add(new_humanoid_obj); all_sprites.add(new_humanoid_obj)
    
    # --- Spawn Landers for the current level ---
    num_landers_base = 10
    num_landers_increase = 4
    num_landers_for_level = num_landers_base + (current_level - 1) * num_landers_increase
    lander_speed_multiplier = 1.0 + (current_level - 1) * 0.08

    for _ in range(num_landers_for_level):
        lander_x = get_random_spawn_x()
        lander_y = random.uniform(30, 180) # Spawn near top of screen
        lander_obj = Lander((lander_x, lander_y), lander_speed_multiplier) # Renamed
        landers.add(lander_obj); all_sprites.add(lander_obj)

# --- Main Game Loop ---
# (Main game loop structure remains the same, calls the modified setup_level)
setup_level(level, is_new_game=True) # Initial setup for Level 1
initial_humanoids_count = len(humanoids) # Set initial count after first setup
game_state = GAME_STATE_START
pygame.USEREVENT_GAME_OVER = pygame.USEREVENT + 1
level_bonus_info = None

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
            if event.key == pygame.K_SPACE:
                if game_state == GAME_STATE_START:
                    level = 1; score = 0; lives = 3; # Full reset for start screen
                    setup_level(level, is_new_game=True);
                    initial_humanoids_count = len(humanoids) # Set count for first level
                    game_state = GAME_STATE_LEVEL_TRANSITION;
                    level_transition_timer = pygame.time.get_ticks();
                    level_bonus_info = None
                elif game_state == GAME_STATE_GAME_OVER:
                    level = 1; score = 0; lives = 3; # Full reset for game over
                    setup_level(level, is_new_game=True);
                    initial_humanoids_count = len(humanoids) # Set count for first level
                    game_state = GAME_STATE_LEVEL_TRANSITION;
                    level_transition_timer = pygame.time.get_ticks();
                    level_bonus_info = None
        if event.type == pygame.USEREVENT_GAME_OVER:
             game_state = GAME_STATE_GAME_OVER

    current_time = pygame.time.get_ticks()

    if game_state == GAME_STATE_PLAYING:
        keys = pygame.key.get_pressed()
        player.handle_input(keys)

        player.update()
        landers.update(humanoids, terrain, player.position)
        humanoids.update(terrain)
        lasers.update(terrain, camera_x)
        enemy_lasers.update(terrain, camera_x)
        explosions.update()

        if not player.is_destroyed:
            camera_x = (player.position.x - WIDTH / 2)
            # camera_x = max(0, min(camera_x, WORLD_WIDTH - WIDTH)) # Optional clamping

        if not player.is_destroyed and not player.invulnerable:
            collided_lander = pygame.sprite.spritecollideany(player, landers)
            if collided_lander:
                collided_lander.destroy(release_humanoid=True)
                player.crash()
            if pygame.sprite.spritecollide(player, enemy_lasers, True): # True to kill lasers on impact
                 player.crash()
            player_bottom_y = player.position.y + player.max_radius * 0.8
            terrain_y_at_player = terrain.get_height_at(player.position.x)
            if player_bottom_y >= terrain_y_at_player:
                 player.crash()

        lander_hits = pygame.sprite.groupcollide(lasers, landers, True, False) # Lasers die, landers don't (handled manually)
        for laser_hit, hit_landers_list in lander_hits.items(): # laser_hit is the key (laser sprite)
            for lander_that_was_hit in hit_landers_list: # lander_that_was_hit is value from list
                # score += 150 # Score is now added to level_score
                level_score += 150
                lander_that_was_hit.destroy(release_humanoid=True)

        for laser_sprite in lasers: # Renamed to avoid conflict
            collided_humanoids_list = pygame.sprite.spritecollide(laser_sprite, humanoids, False) # Don't kill humanoid yet
            for humanoid_collided in collided_humanoids_list: # Renamed
                if humanoid_collided.is_captured:
                    humanoid_collided.is_captured = False; humanoid_collided.is_falling = True
                    # score += 500 # Score is now added to level_score
                    level_score += 500
                    laser_sprite.kill(); # Kill the laser that made the rescue
                    break # Laser can only rescue one per frame/hit

        if game_state == GAME_STATE_PLAYING and not player.is_destroyed and not landers: # Level Clear
            humanoids_saved_count = len(humanoids) # Renamed
            saved_percent = (humanoids_saved_count / initial_humanoids_count) * 100 if initial_humanoids_count > 0 else 100
            bonus_multiplier = saved_percent / 100.0
            level_bonus = int(level_score * bonus_multiplier)
            score += level_score # Add base score from this level
            score += level_bonus # Add bonus to total score
            level_bonus_info = (saved_percent, level_bonus)
            level += 1
            game_state = GAME_STATE_LEVEL_TRANSITION
            level_transition_timer = current_time

        screen.fill(BLACK)
        star_layer_far.draw(screen, camera_x)
        star_layer_near.draw(screen, camera_x)
        terrain.draw(screen, camera_x)
        for sprite_to_draw in all_sprites: # Renamed
            sprite_to_draw.draw(screen, camera_x)
        draw_hud(screen)
        draw_minimap(screen, player, landers, humanoids)

    elif game_state == GAME_STATE_PLAYER_DIED:
        explosions.update()
        if current_time - respawn_timer > RESPAWN_DELAY:
            respawn_x = (camera_x + WIDTH / 2) % WORLD_WIDTH
            respawn_y = HEIGHT * 0.4
            player.respawn((respawn_x, respawn_y))
            game_state = GAME_STATE_PLAYING
        screen.fill(BLACK)
        star_layer_far.draw(screen, camera_x)
        star_layer_near.draw(screen, camera_x)
        terrain.draw(screen, camera_x)
        for sprite_obj in all_sprites:
             if sprite_obj != player:
                 sprite_obj.draw(screen, camera_x)
        for explosion_particle in explosions:
             explosion_particle.draw(screen, camera_x)
        draw_hud(screen)

    elif game_state == GAME_STATE_LEVEL_TRANSITION:
        screen.fill(BLACK)
        if level_bonus_info:
            draw_level_transition_screen(screen, level, level_bonus_info[0], level_bonus_info[1])
        else:
            draw_level_transition_screen(screen, level) # For first level start
        
        if current_time - level_transition_timer > LEVEL_TRANSITION_DURATION:
            setup_level(level, is_new_game=False) # Setup new level (enemies, humanoids)
            initial_humanoids_count = len(humanoids) # Crucial: Set count *after* new humanoids are spawned
            
            respawn_x = (camera_x + WIDTH / 2) % WORLD_WIDTH # Respawn player in current view
            respawn_y = HEIGHT * 0.4
            player.respawn((respawn_x, respawn_y)) # Player appears, invulnerable
            
            level_bonus_info = None # Clear bonus info
            game_state = GAME_STATE_PLAYING

    elif game_state == GAME_STATE_START:
        screen.fill(BLACK); draw_start_screen(screen)

    elif game_state == GAME_STATE_GAME_OVER:
        explosions.update()
        screen.fill(BLACK)
        star_layer_far.draw(screen, camera_x);
        star_layer_near.draw(screen, camera_x);
        terrain.draw(screen, camera_x)
        for explosion_particle in explosions:
             explosion_particle.draw(screen, camera_x)
        draw_game_over_screen(screen)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()