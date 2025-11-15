const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

let width = window.innerWidth;
let height = window.innerHeight;
canvas.width = width;
canvas.height = height;

window.addEventListener('resize', () => {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
});

class Vector2 {
    constructor(x = 0, y = 0) {
        this.x = x;
        this.y = y;
    }

    add(other) {
        return new Vector2(this.x + other.x, this.y + other.y);
    }

    subtract(other) {
        return new Vector2(this.x - other.x, this.y - other.y);
    }

    multiply(scalar) {
        return new Vector2(this.x * scalar, this.y * scalar);
    }

    magnitude() {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }

    normalize() {
        const mag = this.magnitude();
        if (mag === 0) {
            return new Vector2();
        }
        return new Vector2(this.x / mag, this.y / mag);
    }
}

class GameObject {
    constructor(position, color, points = []) {
        this.position = position;
        this.color = color;
        this.base_points = points;
        this.angle = 0;
        this.velocity = new Vector2(0, 0);
        this.max_radius = 0;
        if (this.base_points && this.base_points.length > 0) {
            this.max_radius = Math.max(...this.base_points.map(p => Math.hypot(p.x, p.y)));
        }
        this.alive = true;
        this.visible = true;
    }

    get_transformed_points() {
        const transformed_points = [];
        if (this.base_points) {
            const cos_a = Math.cos(this.angle);
            const sin_a = Math.sin(this.angle);
            for (const point of this.base_points) {
                const rotated_x = point.x * cos_a - point.y * sin_a;
                const rotated_y = point.x * sin_a + point.y * cos_a;
                const world_x = this.position.x + rotated_x;
                const world_y = this.position.y + rotated_y;
                transformed_points.push({ x: world_x, y: world_y });
            }
        }
        return transformed_points;
    }

    draw(ctx, camera_offset_x) {
        if (!this.visible) return;
        const world_points = this.get_transformed_points();
        if (world_points.length === 0) return;

        const screen_points = world_points.map(p => ({ x: p.x - camera_offset_x, y: p.y }));

        this._draw_polygon_if_visible(ctx, screen_points);

        const world_x_coords = world_points.map(p => p.x);
        const min_world_x = Math.min(...world_x_coords);
        const max_world_x = Math.max(...world_x_coords);

        if (min_world_x < this.max_radius * 2 && camera_offset_x > WORLD_WIDTH - width) {
            const wrapped_screen_points = screen_points.map(p => ({ x: p.x + WORLD_WIDTH, y: p.y }));
            this._draw_polygon_if_visible(ctx, wrapped_screen_points);
        } else if (max_world_x > WORLD_WIDTH - this.max_radius * 2 && camera_offset_x < width) {
            const wrapped_screen_points = screen_points.map(p => ({ x: p.x - WORLD_WIDTH, y: p.y }));
            this._draw_polygon_if_visible(ctx, wrapped_screen_points);
        }
    }

    _draw_polygon_if_visible(ctx, screen_points) {
        if (screen_points.length < 2) return;

        const min_x = Math.min(...screen_points.map(p => p.x));
        const max_x = Math.max(...screen_points.map(p => p.x));
        const min_y = Math.min(...screen_points.map(p => p.y));
        const max_y = Math.max(...screen_points.map(p => p.y));

        if (max_x > 0 && min_x < width && max_y > 0 && min_y < height) {
            ctx.strokeStyle = this.color;
            ctx.lineWidth = this instanceof Laser ? 2 : 1;
            ctx.beginPath();
            ctx.moveTo(screen_points[0].x, screen_points[0].y);
            for (let i = 1; i < screen_points.length; i++) {
                ctx.lineTo(screen_points[i].x, screen_points[i].y);
            }
            if (screen_points.length >= 3) {
                ctx.closePath();
            }
            ctx.stroke();
        }
    }

    update() {
        this.position = this.position.add(this.velocity);
        this.position.x = (this.position.x + WORLD_WIDTH) % WORLD_WIDTH;
    }

    destroy(release_humanoid = true) {
        this.alive = false;
        for (let i = 0; i < 20; i++) {
            explosions.push(new ExplosionParticle(this.position, this.color));
        }
        if (this instanceof Lander && this.target_humanoid && this.target_humanoid.is_captured) {
            this.target_humanoid.is_captured = false;
            this.target_humanoid.is_falling = true;
        }
    }
}

class Player extends GameObject {
    constructor(position) {
        const points_right = [ {x: 15, y: 0}, {x: -12, y: -8}, {x: -7, y: 0}, {x: -12, y: 8} ];
        super(position, 'green', points_right);
        this.points_left = points_right.map(p => ({x: -p.x, y: p.y}));
        this.facing_direction = 1;
        this.acceleration = 0.28;
        this.friction = 0.95;
        this.max_speed = 7;
        this.fire_cooldown = 180;
        this.can_fire = true;
        this.fire_cooldown_timer = 0;
        this.accel = new Vector2(0, 0);
        this.is_destroyed = false;
        this.invulnerable = false;
        this.invulnerable_timer = 0;
        this.INVULNERABLE_DURATION = 1500;
        this.visible = true;
    }

    handle_input(keys) {
        if (this.is_destroyed) return;
        this.accel = new Vector2(0, 0);
        let new_facing_direction = this.facing_direction;

        if (keys['ArrowLeft'] || keys['left-btn']) { this.accel.x = -this.acceleration; new_facing_direction = -1; }
        if (keys['ArrowRight'] || keys['right-btn']) { this.accel.x = this.acceleration; new_facing_direction = 1; }
        if (keys['ArrowUp'] || keys['up-btn']) { this.accel.y = -this.acceleration; }
        if (keys['ArrowDown'] || keys['down-btn']) { this.accel.y = this.acceleration; }

        if (new_facing_direction !== this.facing_direction) {
            this.facing_direction = new_facing_direction;
            this.base_points = this.facing_direction === -1 ? this.points_left : this.points_right;
            this.max_radius = Math.max(...this.base_points.map(p => Math.hypot(p.x, p.y)));
        }

        if ((keys['Space'] || keys['fire-btn']) && this.can_fire) {
             this.fire();
             this.can_fire = false;
             this.fire_cooldown_timer = Date.now();
        }
    }

    fire() {
        const laser_start_offset = new Vector2(18 * this.facing_direction, 0);
        const laser_position = this.position.add(laser_start_offset);
        laser_position.x = (laser_position.x + WORLD_WIDTH) % WORLD_WIDTH;

        const laser_speed = 14;
        const base_laser_velocity = new Vector2(laser_speed * this.facing_direction, 0);
        const laser_velocity = base_laser_velocity.add(this.velocity.multiply(0.5));

        lasers.push(new Laser(laser_position, 'cyan', laser_velocity));
    }

    update() {
        const current_time = Date.now();
        if (this.is_destroyed) return;

        if (this.invulnerable) {
            if (current_time - this.invulnerable_timer > this.INVULNERABLE_DURATION) {
                this.invulnerable = false;
                this.visible = true;
            } else {
                this.visible = Math.floor(current_time / 100) % 2 === 0;
            }
        }

        this.velocity = this.velocity.add(this.accel);
        this.velocity = this.velocity.multiply(this.friction);
        if (this.velocity.magnitude() > this.max_speed) {
            this.velocity = this.velocity.normalize().multiply(this.max_speed);
        }

        super.update();

        if (this.position.y > height - this.max_radius) {
            this.position.y = height - this.max_radius;
            this.velocity.y = 0;
        } else if (this.position.y < this.max_radius) {
            this.position.y = this.max_radius;
            this.velocity.y = 0;
        }

        if (!this.can_fire && current_time - this.fire_cooldown_timer > this.fire_cooldown) {
             this.can_fire = true;
        }
    }

    crash() {
        if (!this.invulnerable) {
            this.is_destroyed = true;
            lives--;
            this.destroy(false);
            if (lives > 0) {
                setTimeout(() => this.respawn(), 2000);
            } else {
                // Game over
            }
        }
    }

    respawn() {
        this.position = new Vector2(width / 2, height / 2);
        this.velocity = new Vector2(0, 0);
        this.is_destroyed = false;
        this.invulnerable = true;
        this.invulnerable_timer = Date.now();
    }
}

class Laser extends GameObject {
    constructor(position, color, velocity) {
        const points = [{x: 0, y: 0}, {x: 12, y: 0}];
        super(position, color, points);
        this.velocity = velocity;
        this.lifetime = 60; // 1 second lifetime at 60 FPS
        if (velocity.magnitude() > 0) {
            this.angle = Math.atan2(velocity.y, velocity.x);
        }
    }

    update() {
        super.update();
        this.lifetime -= 1;
        if (this.lifetime <= 0) {
            this.destroy();
        }
    }

    destroy() {
        this.alive = false;
    }
}

class EnemyLaser extends Laser {
    constructor(position, color, velocity) {
        super(position, color, velocity);
        this.base_points = [{x: 0, y: -3}, {x: 0, y: 3}];
    }
}

class ExplosionParticle extends GameObject {
    constructor(position, base_color) {
        super(position, base_color);
        const speed = Math.random() * 4 + 2;
        const angle = Math.random() * 2 * Math.PI;
        this.velocity = new Vector2(Math.cos(angle), Math.sin(angle)).multiply(speed);
        this.lifetime = Math.random() * 20 + 20;
        this.start_radius = Math.random() * 2 + 2;
        this.radius = this.start_radius;
        this.color = ['red', 'orange', 'yellow', 'white'][Math.floor(Math.random() * 4)];
    }

    update() {
        super.update();
        this.lifetime -= 1;
        this.radius = this.start_radius * (this.lifetime / 40.0);
        if (this.lifetime <= 0 || this.radius < 1) {
            this.destroy();
        }
    }

    draw(ctx, camera_offset_x) {
        let screen_x = this.position.x - camera_offset_x;
        const screen_y = this.position.y;

        if (this.radius >= 1) {
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(screen_x, screen_y, this.radius, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

class Humanoid extends GameObject {
    constructor(position) {
        const points = [ {x: 0, y: -8}, {x: 0, y: 0}, {x: -4, y: 5}, {x: 0, y: 0}, {x: 4, y: 5}, {x: 0, y: 0}, {x: -5, y: -4}, {x: 0, y: 0}, {x: 5, y: -4} ];
        super(position, 'white', points);
        this.is_captured = false;
        this.is_falling = false;
        this.fall_speed = 2.5;
    }

    update(terrain) {
        if (this.is_captured) {
            // Logic handled by Lander
        } else if (this.is_falling) {
            this.velocity.y = this.fall_speed;
            const terrain_y = terrain.get_height_at(this.position.x);
            if (this.position.y >= terrain_y - this.max_radius) {
                this.position.y = terrain_y - this.max_radius;
                this.is_falling = false;
                this.velocity.y = 0;
            }
        } else {
            const terrain_y = terrain.get_height_at(this.position.x);
            this.position.y = terrain_y - this.max_radius;
        }
        super.update();
    }
}

class Lander extends GameObject {
    constructor(position, speed_multiplier = 1.0) {
        const points = [ {x: -8, y: 8}, {x: -10, y: 0}, {x: -8, y: -8}, {x: 8, y: -8}, {x: 10, y: 0}, {x: 8, y: 8}, {x: 5, y: 8}, {x: 0, y: 12}, {x: -5, y: 8} ];
        super(position, 'red', points);
        this.state = "descending";
        this.target_humanoid = null;
        this.seek_speed = 1.0 * speed_multiplier;
        this.descent_speed = 0.5 * speed_multiplier;
        this.fire_timer = Math.random() * 150 + 90;
    }

    update(humanoids, terrain) {
        if (this.state === "descending") {
            this.velocity.y = this.descent_speed;
            if (!this.target_humanoid && Math.random() < 0.02) {
                this.find_target(humanoids);
            }
            if (this.target_humanoid) {
                this.state = "seeking";
            }
        } else if (this.state === "seeking") {
            if (this.target_humanoid && !this.target_humanoid.is_captured) {
                const direction = this.target_humanoid.position.subtract(this.position).normalize();
                this.velocity = direction.multiply(this.seek_speed);
                if (this.position.subtract(this.target_humanoid.position).magnitude() < 20) {
                    this.state = "capturing";
                    this.target_humanoid.is_captured = true;
                }
            } else {
                this.find_target(humanoids);
                this.state = "descending";
            }
        } else if (this.state === "capturing") {
            if (this.target_humanoid && this.target_humanoid.is_captured) {
                this.target_humanoid.position = this.position.add(new Vector2(0, this.max_radius + 5));
                this.velocity.y = -this.seek_speed;
            } else {
                this.state = "descending";
            }
        }

        this.fire_timer--;
        if (this.fire_timer <= 0) {
            this.fire();
            this.fire_timer = Math.random() * 150 + 90;
        }

        const terrain_y = terrain.get_height_at(this.position.x);
        if (this.position.y > terrain_y - this.max_radius) {
            this.position.y = terrain_y - this.max_radius;
            this.velocity.y = 0;
        }

        super.update();
    }

    find_target(humanoids) {
        let min_dist = Infinity;
        let target = null;
        for (const humanoid of humanoids) {
            if (!humanoid.is_captured) {
                const dist = this.position.subtract(humanoid.position).magnitude();
                if (dist < min_dist) {
                    min_dist = dist;
                    target = humanoid;
                }
            }
        }
        this.target_humanoid = target;
    }

    fire() {
        const laser_velocity = new Vector2(0, 5);
        enemy_lasers.push(new EnemyLaser(this.position, 'red', laser_velocity));
    }
}

class Terrain {
    constructor(world_width, screen_height, segment_length = 25) {
        this.world_width = world_width;
        this.screen_height = screen_height;
        this.segment_length = segment_length;
        this.points = this._generate_terrain();
    }

    _generate_terrain() {
        const points = [];
        let x = 0;
        let y = this.screen_height * 0.80;
        const min_y = this.screen_height * 0.50;
        const max_y = this.screen_height - 60;
        let slope = 0;
        while (x < this.world_width) {
            points.push({ x, y: Math.round(y) });
            const change_type = Math.random();
            if (change_type < 0.05) {
                slope = 0;
            } else if (change_type < 0.5) {
                const slope_change = Math.random() * 20 - 10;
                slope += slope_change;
            }
            y += slope;
            y = Math.max(min_y, Math.min(y, max_y));
            x += this.segment_length;
        }
        points.push({ x: this.world_width, y: points[0].y });
        return points;
    }

    get_height_at(world_x) {
        world_x = (world_x + this.world_width) % this.world_width;
        const index = Math.floor(world_x / this.segment_length);
        const p1 = this.points[index];
        const p2 = this.points[index + 1];
        if (!p1 || !p2) return this.screen_height;

        const t = (world_x - p1.x) / (p2.x - p1.x);
        return p1.y + t * (p2.y - p1.y);
    }

    draw(ctx, camera_offset_x) {
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (const point of this.points) {
            ctx.lineTo(point.x - camera_offset_x, point.y);
        }
        ctx.stroke();
    }
}

class ParallaxLayer {
    constructor(world_width, num_elements, color, size_range, scroll_factor) {
        this.elements = [];
        this.color = color;
        this.scroll_factor = scroll_factor;
        for (let i = 0; i < num_elements; i++) {
            this.elements.push({
                x: Math.random() * world_width,
                y: Math.random() * height,
                size: Math.random() * (size_range[1] - size_range[0]) + size_range[0]
            });
        }
    }

    draw(ctx, camera_offset_x) {
        ctx.fillStyle = this.color;
        for (const element of this.elements) {
            const parallax_cam_x = camera_offset_x * this.scroll_factor;
            let screen_x = (element.x - parallax_cam_x + WORLD_WIDTH) % WORLD_WIDTH;
            ctx.beginPath();
            ctx.arc(screen_x, element.y, element.size, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

const WORLD_WIDTH = width * 3;
const player = new Player(new Vector2(width / 2, height / 2));
const terrain = new Terrain(WORLD_WIDTH, height);
const star_layer_far = new ParallaxLayer(WORLD_WIDTH, 200, 'gray', [0, 1], 0.1);
const star_layer_near = new ParallaxLayer(WORLD_WIDTH, 100, 'white', [1, 2], 0.3);

let lasers = [];
let enemy_lasers = [];
let humanoids = [];
let landers = [];
let explosions = [];
let camera_x = 0;
let score = 0;
let lives = 3;
let level = 1;
const keys = {};

function setup_level(level) {
    lasers = [];
    enemy_lasers = [];
    humanoids = [];
    landers = [];
    explosions = [];

    for (let i = 0; i < 10; i++) {
        const x = Math.random() * WORLD_WIDTH;
        const y = terrain.get_height_at(x);
        humanoids.push(new Humanoid(new Vector2(x, y)));
    }

    for (let i = 0; i < 5 + level * 2; i++) {
        const x = Math.random() * WORLD_WIDTH;
        const y = Math.random() * height / 2;
        landers.push(new Lander(new Vector2(x, y), 1 + level * 0.1));
    }
}

function check_collisions() {
    // Player lasers with landers
    for (const laser of lasers) {
        for (const lander of landers) {
            if (laser.position.subtract(lander.position).magnitude() < lander.max_radius) {
                laser.destroy();
                lander.destroy();
                score += 150;
            }
        }
    }

    // Enemy lasers with player
    for (const laser of enemy_lasers) {
        if (laser.position.subtract(player.position).magnitude() < player.max_radius) {
            laser.destroy();
            player.crash();
        }
    }

    // Player with landers
    for (const lander of landers) {
        if (player.position.subtract(lander.position).magnitude() < player.max_radius + lander.max_radius) {
            lander.destroy();
            player.crash();
        }
    }
}

window.addEventListener('keydown', e => keys[e.code] = true);
window.addEventListener('keyup', e => keys[e.code] = false);

document.getElementById('left-btn').addEventListener('touchstart', () => keys['left-btn'] = true);
document.getElementById('left-btn').addEventListener('touchend', () => keys['left-btn'] = false);
document.getElementById('right-btn').addEventListener('touchstart', () => keys['right-btn'] = true);
document.getElementById('right-btn').addEventListener('touchend', () => keys['right-btn'] = false);
document.getElementById('up-btn').addEventListener('touchstart', () => keys['up-btn'] = true);
document.getElementById('up-btn').addEventListener('touchend', () => keys['up-btn'] = false);
document.getElementById('down-btn').addEventListener('touchstart', () => keys['down-btn'] = true);
document.getElementById('down-btn').addEventListener('touchend', () => keys['down-btn'] = false);
document.getElementById('fire-btn').addEventListener('touchstart', () => keys['fire-btn'] = true);
document.getElementById('fire-btn').addEventListener('touchend', () => keys['fire-btn'] = false);

function gameLoop() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, width, height);

    player.handle_input(keys);

    if (!player.is_destroyed) {
        player.update();
    }

    lasers.forEach(l => l.update());
    enemy_lasers.forEach(l => l.update());
    humanoids.forEach(h => h.update(terrain));
    landers.forEach(l => l.update(humanoids, terrain));
    explosions.forEach(e => e.update());

    check_collisions();

    lasers = lasers.filter(l => l.alive);
    enemy_lasers = enemy_lasers.filter(l => l.alive);
    landers = landers.filter(l => l.alive);
    humanoids = humanoids.filter(h => h.alive);
    explosions = explosions.filter(e => e.alive);

    if (landers.length === 0) {
        level++;
        setup_level(level);
    }

    camera_x = player.position.x - width / 2;

    star_layer_far.draw(ctx, camera_x);
    star_layer_near.draw(ctx, camera_x);
    terrain.draw(ctx, camera_x);

    if (!player.is_destroyed) {
        player.draw(ctx, camera_x);
    }

    lasers.forEach(l => l.draw(ctx, camera_x));
    enemy_lasers.forEach(l => l.draw(ctx, camera_x));
    humanoids.forEach(h => h.draw(ctx, camera_x));
    landers.forEach(l => l.draw(ctx, camera_x));
    explosions.forEach(e => e.draw(ctx, camera_x));

    // Draw HUD
    ctx.fillStyle = 'white';
    ctx.font = '24px Arial';
    ctx.fillText(`Score: ${score}`, 10, 30);
    ctx.fillText(`Lives: ${lives}`, 10, 60);
    ctx.fillText(`Level: ${level}`, 10, 90);

    requestAnimationFrame(gameLoop);
}

setup_level(level);
gameLoop();
