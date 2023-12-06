"""
@file twomix_gravity.py

Handles generating a mixed set of a simple static velocity set (left to right, same initial velocities) alongside
a single gravity
"""
import os
import pygame
import random
import tarfile
import numpy as np
import pymunk.pygame_util
import matplotlib.pyplot as plt

from tqdm import tqdm


class BallBox:
    def __init__(self, dt=0.2, res=(32, 32), init_pos=(3, 3), init_std=0, wall=None, gravity=(0.0, 0.0), ball_color="white"):
        pygame.init()

        self.ball_color = ball_color

        self.dt = dt
        self.res = res
        if os.environ.get('SDL_VIDEODRIVER', '') == 'dummy':
            pygame.display.set_mode(res, 0, 24)
            self.screen = pygame.Surface(res, pygame.SRCCOLORKEY, 24)
            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, res[0], res[1]), 0)
        else:
            self.screen = pygame.display.set_mode(res, 0, 24)
        self.gravity = gravity
        self.initial_position = init_pos
        self.initial_std = init_std
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()
        self.wall = wall
        self.static_lines = None

        self.dd = 2

    def _clear(self):
        self.screen.fill((0.3176, 0.3451, 0.3647))

    def create_ball(self, radius=3):
        inertia = pymunk.moment_for_circle(1, 0, radius, (0, 0))
        body = pymunk.Body(1, inertia)
        position = np.array(self.initial_position) + self.initial_std * np.random.normal(size=(2,))
        position = np.clip(position, self.dd + radius + 1, self.res[0]-self.dd-radius-1)
        position = position.tolist()
        body.position = position

        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 1.0

        shape.color = pygame.color.THECOLORS[self.ball_color]
        return shape

    def fire(self, angle=50, velocity=20, radius=3):
        speedX = velocity * np.cos(angle * np.pi / 180)
        speedY = velocity * np.sin(angle * np.pi / 180)

        ball = self.create_ball(radius)
        ball.body.velocity = (speedX, speedY)

        self.space.add(ball, ball.body)
        return ball

    def run(self, iterations=20, sequences=500, angle_limits=(0, 360), velocity_limits=(10, 25), radius=3,
            flip_gravity=None, save=None, filepath='../../data/balls.npz', delay=None):
        if save:
            images = np.empty((sequences, iterations, self.res[0], self.res[1]), dtype=np.float32)
            state = np.empty((sequences, iterations, 2), dtype=np.float32)

        dd = 0
        self.static_lines = [pymunk.Segment(self.space.static_body, (-1, -1), (-1, self.res[1]-dd), 0.0),
                             pymunk.Segment(self.space.static_body, (-1, -1), (self.res[0]-dd, -1), 0.0),
                             pymunk.Segment(self.space.static_body, (self.res[0] - dd, self.res[1] - dd),
                                            (-1, self.res[1]-dd), 0.0),
                             pymunk.Segment(self.space.static_body, (self.res[0] - dd, self.res[1] - dd),
                                            (self.res[0]-dd, -1), 0.0)]
        for line in self.static_lines:
            line.elasticity = 1.0

            if self.ball_color == "white2":
                line.color = pygame.color.THECOLORS["white"]
            else:
                line.color = pygame.color.THECOLORS[self.ball_color]
        # self.space.add(self.static_lines)

        for sl in self.static_lines:
            self.space.add(sl)

        for s in range(sequences):
            if s % 100 == 0:
                print(s)

            angle = np.random.uniform(*angle_limits)
            velocity = np.random.uniform(*velocity_limits)
            # controls[:, s] = np.array([angle, velocity])
            ball = self.fire(angle, velocity, radius)
            for i in range(iterations):
                self._clear()
                self.space.debug_draw(self.draw_options)
                self.space.step(self.dt)
                pygame.display.flip()

                if delay:
                    self.clock.tick(delay)

                if save == 'png':
                    pygame.image.save(self.screen, os.path.join(filepath, "bouncing_balls_%02d_%02d.png" % (s, i)))
                elif save == 'npz':
                    images[s, i] = pygame.surfarray.array2d(self.screen).swapaxes(1, 0).astype(np.float32) / (2**24 - 1)
                    state[s, i] = list(ball.body.velocity) # list(ball.body.position) + # Note that this is done for compatibility with the combined dataset

            # Remove the ball and the wall from the space
            self.space.remove(ball, ball.body)

        return images, state


if __name__ == '__main__':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    # Parameters of generation, resolution and number of samples
    scale = 1
    timesteps = 75
    training_size = 25000
    validation_size = 5000
    testing_size = 5000

    base_dir = f"bouncingball_{training_size}samples_{timesteps}steps/"

    # Arrays to hold sets
    train_images, train_states = [], []
    test_images, test_states = [], []
    np.random.seed(1234)

    # Generate the data sequences
    cannon = BallBox(dt=0.5, res=(32*scale, 32*scale), init_pos=(16*scale, 16*scale), init_std=8,
                     wall=None, gravity=(0.0, 0.0), ball_color="white")
    i, s = cannon.run(delay=None, iterations=timesteps, sequences=training_size + validation_size + testing_size,
                      radius=4, angle_limits=(0, 360), velocity_limits=(5.0, 10.0), save='npz')

    # Setting the pixels to a uniform background and foreground (for simplicity of training)
    i[i > 0] = 1
    i[i == 0] = 0.32

    # Break into train and test sets, adding in generic labels
    train_images = i[:training_size]
    train_states = s[:training_size]
    train_classes = np.full([train_images.shape[0], 1], fill_value=2)

    val_images = i[training_size:training_size + validation_size]
    val_states = s[training_size:training_size + validation_size]
    val_classes = np.full([val_images.shape[0], 1], fill_value=2)

    test_images = i[training_size + validation_size:]
    test_states = s[training_size + validation_size:]
    test_classes = np.full([test_images.shape[0], 1], fill_value=2)

    print(f"Train - Images: {train_images.shape} | States: {train_states.shape} | Classes: {train_classes.shape}")
    print(f"Val - Images: {val_images.shape} | States: {val_states.shape} | Classes: {val_classes.shape}")
    print(f"Test - Images: {test_images.shape} | States: {test_states.shape} | Classes: {test_classes.shape}")

    # Make sure all directories are made beforehand
    if not os.path.exists(f"{base_dir}/"):
        os.mkdir(f"{base_dir}/")

    # Permute the sets and states together
    p = np.random.permutation(train_images.shape[0])
    train_images, train_classes, train_states = train_images[p], train_classes[p], train_states[p]

    p = np.random.permutation(val_images.shape[0])
    val_images, val_classes, val_states = val_images[p], val_classes[p], val_states[p]

    p = np.random.permutation(test_images.shape[0])
    test_images, test_classes, test_states = test_images[p],test_classes[p], test_states[p]

    # Save sets
    np.savez(os.path.abspath(f"{base_dir}/train.npz"), image=train_images, state=train_states, label=train_classes)
    np.savez(os.path.abspath(f"{base_dir}/val.npz"), image=val_images, state=val_states, label=val_classes)
    np.savez(os.path.abspath(f"{base_dir}/test.npz"), image=test_images, state=test_states, label=test_classes)
