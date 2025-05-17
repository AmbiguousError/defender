# Vector Defender - Continuous!

A fast-paced, side-scrolling vector shooter inspired by the arcade classic "Defender." Navigate a vast, looping world, protect humanoids from alien abduction, and blast waves of increasingly challenging enemies.

## Story

Brave pilot, the fate of the last humanoids rests in your hands! Alien Landers are descending upon your world, intent on abducting the vulnerable population. Take control of your advanced fighter ship, fend off the extraterrestrial threat, and ensure the survival of as many humanoids as possible. Each level brings new challenges and more determined foes.

## Gameplay

* **Pilot Your Ship:** Use the arrow keys to navigate the continuous, wrapping game world.
* **Laser Fire:** Press the Spacebar to unleash a volley of laser fire against enemy ships.
* **Protect Humanoids:** Prevent Landers from capturing and abducting the humanoids scattered across the terrain.
    * Landers will descend, seek out humanoids, and attempt to carry them off the top of the screen.
    * Destroying a Lander carrying a humanoid will release the captive, allowing them to fall back to safety (if they don't fall too far!).
    * You can also shoot a captured humanoid (while it's being carried) to free it – a risky but sometimes necessary maneuver!
* **Enemy Variety:** Face off against increasingly difficult Landers that become faster and more aggressive as you progress through levels. Beware of their retaliatory laser fire!
* **Level Progression:** Clear all Landers in a level to advance. Your performance, based on humanoids saved and score accumulated during the level, will contribute to a level completion bonus.
* **Continuous World:** The game world seamlessly wraps around. Fly far enough in one direction, and you'll find yourself back where you started, but the action never stops!
* **Dynamic Terrain:** Navigate a procedurally generated, varied terrain that provides both cover and obstacles.
* **Lives and Score:** You have a limited number of lives. Colliding with enemies, enemy fire, or the terrain will result in the loss of a life. Strive for a high score by destroying enemies and saving humanoids.

## Features

* Classic vector graphics aesthetic.
* Smooth, responsive controls.
* Endless, looping gameplay across a large world.
* Dynamic parallax scrolling for a sense of depth.
* Procedurally generated sound effects for lasers and explosions.
* Challenging AI for enemy Landers.
* Score tracking, multiple lives, and level-based difficulty scaling.
* Mini-map to help navigate the expansive world and keep track of humanoids and enemies.

## How to Play

1.  **Launch the Game:**
    * If you have Python and Pygame installed, you can run the `defender.py` script directly:
        ```bash
        python defender.py
        ```
    * **Alternatively, a pre-compiled version for Windows, `defender.exe`, is available in the `dist` folder.** Simply run this executable to play.
2.  **Controls:**
    * **Arrow Keys:** Move Ship (Up, Down, Left, Right)
    * **Spacebar:** Fire Laser
    * **ESC:** Quit Game
3.  **Objective:**
    * Survive as long as possible.
    * Protect the humanoids from abduction.
    * Destroy all Landers to complete a level.
    * Achieve the highest score!

## Development

This game is built using Python and the Pygame library. Key Python libraries used include:

* `pygame`: For graphics, sound, input, and game loop management.
* `numpy`: For generating sound wave data.
* `math`: For vector calculations and rotations.
* `random`: For various randomized game elements (enemy behavior, terrain, etc.).
* `sys`: For system-level operations (like exiting the game).
* `time`: For managing respawn delays and transitions.

## Future Ideas (Optional)

* More enemy types with unique behaviors.
* Player power-ups (e.g., shields, rapid fire, smart bombs).
* More detailed scoring events.
* Persistent high score table.

---

Enjoy defending the last of humanity in **Vector Defender - Continuous!**