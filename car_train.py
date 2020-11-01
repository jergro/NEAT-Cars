import os
import pygame
import pygame.locals as pygl
import numpy as np
import neat

pygame.init()

screen_width = 1200
screen_height = 800

# initialize the display, set a caption, open the race track img, and start the clock.
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Cars")
background = pygame.transform.scale(pygame.image.load('background.png').convert_alpha(),
                                    (screen_width, screen_height))


class Car(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(Car, self).__init__()
        self.x = x
        self.y = y
        self.vel = 0
        self.rot = 90
        self.fitness = 0
        self.is_dead = False
        self.goal = False
        self.img = pygame.transform.scale(pygame.image.load('car.png').convert_alpha(),
                                          (50, 50))
        self.distances = [0, 0, 0, 0, 0]

    def update(self, left, right):#, brake):
        """

        Car goes vroom, car goes skrrrt.

        :return:
        """
        # self.update_fitness(dt)
        if left and self.vel != 0:
            self.rot += 5 + np.sqrt(self.vel)

        if right and self.vel != 0:
            self.rot -= 5 + np.sqrt(self.vel)

        # if brake:
        #     acc = -0.8
        #     self.vel = self.vel + 0.5 * acc
        #     if self.vel <= 0:
        #         self.vel = 0
        #     self.x -= self.vel * np.cos(np.deg2rad(self.rot))
        #     self.y += self.vel * np.sin(np.deg2rad(self.rot))

        acc = 0.8
        self.vel = self.vel + 0.5 * acc
        if self.vel > 10:
            self.vel = 10
        self.x -= self.vel * np.cos(np.deg2rad(self.rot))
        self.y += self.vel * np.sin(np.deg2rad(self.rot))

    def draw(self):
        """

        rotates the car image around its center and blits it.

        :return: None
        """
        copy_img = pygame.transform.rotate(self.img, self.rot)
        img_width = copy_img.get_width()
        img_height = copy_img.get_height()
        screen.blit(copy_img, (self.x - img_width / 2, self.y - img_height / 2))

    def get_mask(self):
        """

        Rotate the car around the center and get the mask.
        calculates the offset to the center of the car.

        :return: mask and both x and y offset.
        """
        copy_img = pygame.transform.rotate(self.img, self.rot)
        img_width = copy_img.get_width()
        img_height = copy_img.get_height()
        offset_x = int(self.x - img_width / 2)
        offset_y = int(self.y - img_height / 2)
        return pygame.mask.from_surface(copy_img), offset_x, offset_y

    def draw_rays(self, rot):
        """

        Ray casting in order for the car to know the distances to the walls.

        :param rot: The rotation for the given ray.
        :return: calculates the distance from the cast ray to the wall.
        """
        ray_surface.fill((0, 0, 0, 0))
        c = np.cos(np.deg2rad(self.rot + rot))
        s = np.sin(np.deg2rad(self.rot + rot))
        flip_x = c > 0
        flip_y = s < 0

        # Flip mask, car position and beam direction to the new coordinate system.
        flipped_mask = flipped_masks[flip_x][flip_y]
        x_start, y_start = map_coordinates(self.x, self.y, flip_x, flip_y)
        x_end, y_end = map_coordinates(self.x - 1000 * c, self.y + 1000 * s, flip_x, flip_y)

        # Draw ray in the new coordinates and get the mask.
        pygame.draw.line(ray_surface, (0, 0, 255), (x_start, y_start), (x_end, y_end))
        beam_mask = pygame.mask.from_surface(ray_surface)

        # check ray overlaps with the racetrack.
        hit = flipped_mask.overlap(beam_mask, (0, 0))

        if hit is not None:
            # flip back to the original coordinate system. calculate distance.
            hx, hy = map_coordinates(hit[0], hit[1], flip_x, flip_y)
            pygame.draw.circle(screen, (0, 255, 0), (hx, hy), 3)
            pygame.draw.line(screen, (0, 0, 255), (self.x, self.y), (hx, hy))
            return ((self.x - hx) ** 2 + (self.y - hy) ** 2) ** 0.5

    def update_fitness(self, dt, step):
        self.fitness += self.vel * dt / 100000
        self.fitness -= dt / 20000
        if 70 < self.x < 180 and 60 < self.y < 150 and not self.goal:
            self.fitness += 100/step
            self.goal = True
        if self.is_dead:
            self.fitness -= 10

def map_coordinates(x, y, flip_x, flip_y):
    """
    Maps coordinates from the normal coordinate system to one thats fliped around either of the axis.

    :param x: x-coordinate in original coordinate system.
    :param y: y-coordinate in original coordinate system.
    :param flip_x: Boolean, if true flip about the midpoint in the x direction.
    :param flip_y: Boolean, if true flip about the midpoint in the y direction.
    :return: the coordinates in the new coordinate system.
    """
    x_after, y_after = x, y
    if flip_x:
        x_after = abs(x - screen_width)
    if flip_y:
        y_after = abs(y - screen_height)
    return x_after, y_after

def get_masks(surface):
    """

    Creates pygame mask objeccts in matrix form. Function only for ease of readability.

    :param surface: pygame surface.
    :return: a list of masks flipped around different axis in matrix form think rotational matrix ish.
    """
    back_mask = pygame.mask.from_surface(surface)
    back_mask_fx = pygame.mask.from_surface(pygame.transform.flip(surface, True, False))
    back_mask_fy = pygame.mask.from_surface(pygame.transform.flip(surface, False, True))
    back_mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(surface, True, True))
    return [[back_mask, back_mask_fy], [back_mask_fx, back_mask_fx_fy]]


# Get the masks of the race track in all rotations.
flipped_masks = get_masks(background)
ray_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)

# time change for each game loop
dt = 0

# start generations
gen = 0


def eval_genomes(genomes, config):
    """
    runs the simulation of the current population of
    cars.
    """
    global gen, dt
    gen += 1

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # car object that uses that network to play
    nets = []
    cars = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(270, 100))
        ge.append(genome)

    clock = pygame.time.Clock()

    # main program loop
    step = 0
    running = True
    while running and len(cars) > 0:
        for event in pygame.event.get():
            if event.type == pygl.KEYDOWN:
                if event.key == pygl.K_ESCAPE:
                    running = False
            elif event.type == pygl.QUIT:
                running = False
                pygame.quit()
                break

        # blit the background (race track)
        screen.fill((255, 255, 255))
        screen.blit(background, (0, 0))

        # blit the car and update it
        # ticks = pygame.time.get_ticks()
        for i, car in enumerate(cars):
            car.draw()

            car.update_fitness(dt, step)
            ge[i].fitness = car.fitness

            # get the mask of the car together with the ofset.
            car_mask, offset_x, offset_y = car.get_mask()

            # Check for overlap. Car with the track. pixel perfect.
            collide = flipped_masks[0][0].overlap(car_mask, (offset_x, offset_y))

            if collide:
                # Draw a circle where the car crashed and kill it.
                # pygame.draw.circle(screen, (255, 0, 0), overlap, 3)
                car.fitness -= 1
                car.is_dead = True
                ge[i].fitness -= 1
                nets.pop(i)
                ge.pop(i)
                cars.pop(i)

            # Create rays for calculating distances.
            if not car.is_dead:
                for j, angle in enumerate([-90, -20, 0, 20, 90]):
                    car.distances[j] = car.draw_rays(angle)

                output = nets[i].activate((car.vel,
                                           car.distances[0],
                                           car.distances[1],
                                           car.distances[2],
                                           car.distances[3],
                                           car.distances[4]))

                car.update(output[0] > 0.5, output[1] > 0.5)#, output[2] > 0.5)

        step += 1
        dt = clock.tick(100)
        pygame.display.flip()

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
