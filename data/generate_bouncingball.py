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
    train_size = 10000
    test_size = 2000

    # Arrays to hold sets
    train_images, train_states = [], []
    test_images, test_states = [], []
    np.random.seed(1234)

    # Generate the first set
    cannon = BallBox(dt=0.2, res=(32*scale, 32*scale), init_pos=(16*scale, 16*scale), init_std=8, wall=None, gravity=(0.0, 0.0), ball_color="white")
    i, s = cannon.run(delay=None, iterations=65, sequences=train_size, radius=4, angle_limits=(0, 360), velocity_limits=(5.0, 10.0), save='npz')
    i[i > 0] = 1
    i[i == 0] = 0.32
    train_images.append(i)
    train_classes = np.full([i.shape[0], 1], fill_value=2)
    train_states.append(s)

    cannon = BallBox(dt=0.2, res=(32*scale, 32*scale), init_pos=(16*scale, 16*scale), init_std=8, wall=None, gravity=(0.0, 0.0), ball_color="white")
    i, s = cannon.run(delay=None, iterations=65, sequences=test_size, radius=4, angle_limits=(0, 360), velocity_limits=(5.0, 10.0), save='npz')
    i[i > 0] = 1
    i[i == 0] = 0.32

    test_images.append(i)
    test_classes = np.full([i.shape[0], 1], fill_value=2)
    test_states.append(s)

    # Concatenate the images together
    train_images = np.concatenate(train_images, axis=0)
    train_states = np.concatenate(train_states, axis=0)
    test_images = np.concatenate(test_images, axis=0)
    test_states = np.concatenate(test_states, axis=0)

    train_states, test_states = train_states[:, np.newaxis, :, :], test_states[:, np.newaxis, :, :]
    print(train_images.shape, train_classes.shape, test_images.shape)
    print(f"Images: {train_images.shape} | States: {train_states.shape} | Classes: {train_classes.shape}")

    train_size = train_images.shape[0]
    test_size = test_images.shape[0]

    # Plot some examples
    def movie_to_frame(images):
        """ Compiles a list of images into one composite frame """
        n_steps, w, h = images.shape
        colors = np.linspace(0.4, 1, n_steps)
        image = np.zeros((w, h))
        for i, color in zip(images, colors):
            image = np.clip(image + i * color, 0, color)
        return image

    base_dir = "hamiltonian/bouncing_ball/bouncingball_10000samples_65steps/"

    # Make sure all directories are made beforehand
    if not os.path.exists(f"{base_dir}/examples/"):
        os.mkdir(f"{base_dir}/examples/")

    if not os.path.exists(f"{base_dir}/train/"):
        os.mkdir(f"{base_dir}/train/")

    if not os.path.exists(f"{base_dir}/test/"):
        os.mkdir(f"{base_dir}/test/")

    if not os.path.exists(f"{base_dir}/train_tars/"):
        os.mkdir(f"{base_dir}/train_tars/")

    if not os.path.exists(f"{base_dir}/test_tars/"):
        os.mkdir(f"{base_dir}/test_tars/")

    # Grab random samples and save stacked samples from sequence
    selected_idx = np.random.choice(train_size, 10, replace=False)
    for idx in selected_idx:
        plt.imshow(movie_to_frame(train_images[idx]), cmap='gray')
        plt.savefig(f'{base_dir}/examples/train_{idx}.png')

    selected_idx = np.random.choice(test_size, 10, replace=False)
    for idx in selected_idx:
        plt.imshow(movie_to_frame(test_images[idx]), cmap='gray')
        plt.savefig(f'{base_dir}/examples/test_{idx}.png')

    # Permute the sets and states together
    p = np.random.permutation(train_images.shape[0])
    train_images, train_classes, train_states = train_images[p], train_classes[p], train_states[p]

    p = np.random.permutation(test_images.shape[0])
    test_images, test_classes, test_states = test_images[p],test_classes[p], test_states[p]

    # Save as individual files
    for idx, (i, c, s) in enumerate(zip(train_images, train_classes, train_states)):
        np.savez(os.path.abspath(f"{base_dir}/train/{idx}.npz"), image=i, x=s, class_id=c)

    for idx, (i, c, s) in enumerate(zip(test_images, test_classes, test_states)):
        np.savez(os.path.abspath(f"{base_dir}/test/{idx}.npz"), image=i, x=s, class_id=c)

    # Tar train into chunks for WebDataset
    file_list = os.listdir(f"{base_dir}/train/")
    random.shuffle(file_list)

    n_shards = 1000
    elements_per_shard = len(file_list) // n_shards
    for n in tqdm(range(n_shards)):
        with tarfile.open(f"{base_dir}/train_tars/{n:0:04}.tar", "w:gz") as tar:
            for file in file_list[n * elements_per_shard: (n + 1) * elements_per_shard]:
                tar.add(f"{base_dir}/train/{file}")

    # Tar test into chunks for WebDataset
    file_list = os.listdir(f"{base_dir}/test/")
    random.shuffle(file_list)

    n_shards = 1000
    elements_per_shard = len(file_list) // n_shards
    for n in tqdm(range(n_shards)):
        with tarfile.open(f"{base_dir}/test_tars/{n:0:04}.tar", "w:gz") as tar:
            for file in file_list[n * elements_per_shard: (n + 1) * elements_per_shard]:
                tar.add(f"{base_dir}/test/{file}")
