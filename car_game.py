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
background = pygame.transform.scale(pygame.image.load("background.png").convert_alpha(), (screen_width, screen_height))
clock = pygame.time.Clock()

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
        self.img = pygame.transform.scale(pygame.image.load("car.png").convert_alpha(), (50, 50))
        self.distances = np.zeros(91)

    def update(self, dt):
        """

        Car goes vroom, car goes skrrrt.

        :return:
        """
        self.update_fitness(dt)
        if pygame.key.get_pressed()[pygl.K_LEFT]:
            self.rot += np.sqrt(self.vel + 5)

        if pygame.key.get_pressed()[pygl.K_RIGHT]:
            self.rot -= np.sqrt(self.vel + 5)

        if pygame.key.get_pressed()[pygl.K_UP]:
            acc = 0.8
            self.vel = self.vel + 0.5 * acc
            if self.vel > 10:
                self.vel = 10
            self.x -= self.vel * np.cos(np.deg2rad(self.rot))
            self.y += self.vel * np.sin(np.deg2rad(self.rot))

        else:
            acc = -0.8
            self.vel = self.vel + 0.5 * acc
            if self.vel <= 0:
                self.vel = 0
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
            pygame.draw.line(screen, (0,0,255), (self.x, self.y), (hx,hy))
            return ((self.x - hx)**2 + (self.y - hy)**2)**0.5

    def update_fitness(self, dt):
        self.fitness += self.vel*dt/100000
        self.fitness -= dt/20000
        if 70 < self.x < 180 and 60 < self.y < 150 and not self.goal:
            self.fitness += 10
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
car = Car(270, 100)

# main program loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygl.KEYDOWN:
            if event.key == pygl.K_ESCAPE:
                running = False
        elif event.type == pygl.QUIT:
            running = False

    # blit the background (race track)
    screen.fill((255, 255, 255))
    screen.blit(background, (0, 0))

    # blit the car and update it
    # ticks = pygame.time.get_ticks()
    car.draw()
    if not car.is_dead:
        car.update(dt)

    # get the mask of the car together with the offset.
    car_mask, offset_x, offset_y = car.get_mask()

    # Check for overlap. Car with the track. pixel perfect.
    overlap = flipped_masks[0][0].overlap(car_mask, (offset_x, offset_y))

    if overlap:
        # Draw a circle where the car crashed and kill it.
        pygame.draw.circle(screen, (255, 0, 0), overlap, 3)
        car.is_dead = True

    # Create rays for calculating distances.
    if not car.is_dead:
        for i, angle in enumerate([-90, -45, 0, 45, 90]):
            car.distances[i] = car.draw_rays(angle)

    dt = clock.tick(30)
    pygame.display.flip()
pygame.quit()
