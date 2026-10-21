import pygame
import random
import numpy as np
import cv2

# Obstacle variants: (width, height, y_top_offset_from_ground)
# y_top_offset is how far ABOVE ground_y the top of the obstacle sits.
CACTUS_VARIANTS = [
    dict(name="cactus_small", w=12, h=25, y_top=25),
    dict(name="cactus_large", w=20, h=35, y_top=35),
    dict(name="cactus_cluster", w=38, h=25, y_top=25),
]

# Birds: two height bands.
#   low_bird  -> overlaps standing AND jumping dino, but NOT ducking dino. Must duck.
#   high_bird -> sits above a standing dino's head. Running under is safe;
#                ducking is also safe; jumping into it is risky. Correct action: do nothing.
BIRD_VARIANTS = [
    dict(name="low_bird", w=22, h=16, y_top=45),
    dict(name="high_bird", w=22, h=16, y_top=70),
]


class HeadlessDinoEnv:
    def __init__(self, width=600, height=150):
        pygame.init()
        self.width = width
        self.height = height
        # Render to an in-memory surface without a physical window
        self.surface = pygame.Surface((self.width, self.height))
        self.ground_y = 120
        self.reset()

    def reset(self):
        self.dino_x = 40
        self.dino_y = self.ground_y - 30
        self.dino_vy = 0
        self.is_jumping = False
        self.is_ducking = False
        self.obstacles = []  # list of dicts: {rect, type}
        self.base_speed = 6.0
        self.spawn_timer = 0
        self.next_spawn_gap = random.randint(45, 90)
        self.score = 0
        self.done = False
        return self._get_obs()

    @property
    def obstacle_speed(self):
        # Difficulty ramp, mirrors the real game's gradual speed-up, capped.
        return min(self.base_speed + self.score * 0.002, 13.0)

    def step(self, action):
        """
        action: 0 = Run, 1 = Jump, 2 = Duck
        """
        if self.done:
            return self._get_obs(), 0, True, {}

        # 1. Action Logic
        if action == 1 and not self.is_jumping:
            self.is_jumping = True
            self.dino_vy = -12  # Jump impulse
            self.is_ducking = False
        elif action == 2 and not self.is_jumping:
            self.is_ducking = True
        else:
            self.is_ducking = False

        # 2. Dino Physics
        if self.is_jumping:
            self.dino_y += self.dino_vy
            self.dino_vy += 0.8  # Gravity
            if self.dino_y >= self.ground_y - 30:
                self.dino_y = self.ground_y - 30
                self.dino_vy = 0
                self.is_jumping = False

        dino_w = 20
        dino_h = 15 if self.is_ducking else 30
        current_dino_y = self.ground_y - 15 if self.is_ducking else self.dino_y
        dino_rect = pygame.Rect(self.dino_x, current_dino_y, dino_w, dino_h)

        # 3. Obstacle Spawning & Movement
        speed = self.obstacle_speed
        self.spawn_timer += 1
        if self.spawn_timer > self.next_spawn_gap:
            if random.random() < 0.5:
                variant = random.choice(CACTUS_VARIANTS)
                obs_type = "cactus"
            else:
                variant = random.choice(BIRD_VARIANTS)
                obs_type = "bird"

            rect = pygame.Rect(
                self.width,
                self.ground_y - variant["y_top"],
                variant["w"],
                variant["h"],
            )
            self.obstacles.append({"rect": rect, "type": obs_type, "name": variant["name"]})
            self.spawn_timer = 0
            self.next_spawn_gap = random.randint(45, 90)

        for obs in self.obstacles:
            obs["rect"].x -= int(round(speed))
            if dino_rect.colliderect(obs["rect"]):
                self.done = True

        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs["rect"].x > -50]

        self.score += 1
        reward = 1.0 if not self.done else -10.0
        return self._get_obs(), reward, self.done, {"next_obstacle": self._get_next_obstacle_info()}

    def _get_next_obstacle_info(self):
        """Helper to guide the data-collection heuristic agent."""
        incoming = [obs for obs in self.obstacles if obs["rect"].x + obs["rect"].w >= self.dino_x]
        if not incoming:
            return {"dist": 9999, "type": None, "y_top": None, "y_bottom": None}
        nearest = min(incoming, key=lambda o: o["rect"].x - self.dino_x)
        r = nearest["rect"]
        return {
            "dist": r.x - self.dino_x,
            "type": nearest["type"],
            "y_top": r.y,
            "y_bottom": r.y + r.h,
        }

    def _get_obs(self):
        # 1. Draw frame to surface (White background, Black shapes)
        self.surface.fill((255, 255, 255))
        pygame.draw.line(self.surface, (0, 0, 0), (0, self.ground_y), (self.width, self.ground_y), 2)

        # Draw Dino
        dino_h = 15 if self.is_ducking else 30
        current_dino_y = self.ground_y - 15 if self.is_ducking else self.dino_y
        pygame.draw.rect(self.surface, (0, 0, 0), [self.dino_x, current_dino_y, 20, dino_h])

        # Draw Obstacles (cacti as rectangles, birds as distinct winged ovals)
        for obs in self.obstacles:
            r = obs["rect"]
            if obs["type"] == "cactus":
                pygame.draw.rect(self.surface, (0, 0, 0), r)
            else:
                pygame.draw.ellipse(self.surface, (0, 0, 0), r)
                mid_y = r.y + r.h // 2
                wing_flap = 6 if (self.score // 5) % 2 == 0 else -6
                pygame.draw.line(self.surface, (0, 0, 0), (r.x, mid_y), (r.x - 8, mid_y + wing_flap), 2)
                pygame.draw.line(self.surface, (0, 0, 0), (r.x + r.w, mid_y), (r.x + r.w + 8, mid_y + wing_flap), 2)

        # 2. Extract pixel array: (W, H, 3) -> Transpose to (H, W, 3)
        img = pygame.surfarray.array3d(self.surface).transpose([1, 0, 2])

        # 3. Grayscale and resize to 64x64
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

        # Return shape: (1, 64, 64) as uint8 (0-255)
        return np.expand_dims(resized, axis=0)
