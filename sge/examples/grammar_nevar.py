import re
import random
import pygame
from tqdm import tqdm
import numpy as np
from pygame.surfarray import make_surface, blit_array
from numpy import array as nparray, zeros, ones
from pygame import key
from pygame.locals import *
from math import isnan, isinf, sqrt
from sge.utilities.mathnumpy import _log_, _div_, _exp_, _sqrt_, protdiv, getcenterdistance, _div_, _sum_, _sub_, \
    _mul_,_sin_,_cos_,_tan_, _sinh_, _cosh_, _xor_, distance_to_point, gaussian


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


class Artist:
    def __init__(self, imgsize=(256, 256)):
        self.imgsize = imgsize
        self._running = False
        self.pause = False
        self.selected = []
        self.surfaces = []
        self.display_surf = None
        self.zoompanel = None

    def zoomsurface(self, surface, panel):
        zoomed = pygame.transform.scale2x(surface)
        panel.image = zoomed


    def render_images(self, pop, img_size, feedback=False, **kwargs):
        images = []
        for ind in tqdm(pop):
            phenotype = ind['phenotype']
            #print(phenotype)
            r_function, g_function, b_function = phenotype.split('\n')
            x, y = np.mgrid[0:img_size[0]:(img_size[0])*1j, 0:img_size[1]:(img_size[1])*1j]
            pointN = np.vstack((x.flatten(), y.flatten())).T * 1.0
            r_function = eval(r_function)
            r = np.apply_along_axis(r_function, 1, pointN) * 1.0
            r[np.where(np.isnan(r))] = 0.0
            r[np.where(np.isinf(r))] = 0.0
            r[np.where(np.abs(r) > 1.0)] /= 255.0
            r[np.where(np.abs(r) < 1.0)] *= 255.0
            if r.shape[1] < 2:
                r = np.hstack((r, np.zeros((r.shape[0], 1))))
            r = r.mean(axis=1)
            r = np.reshape(r, img_size)

            g_function = eval(g_function)
            g = np.apply_along_axis(g_function, 1, pointN) * 1.0
            g[np.where(np.isnan(g))] = 0.0
            g[np.where(np.isinf(g))] = 0.0
            g[np.where(np.abs(g) > 1.0)] /= 255.0
            g[np.where(np.abs(g) < 1.0)] *= 255.0
            if g.shape[1] < 2:
                g = np.hstack((g, np.zeros((g.shape[0], 1))))
            g = g.mean(axis=1)
            g = np.reshape(g, img_size)

            b_function = eval(b_function)
            b = np.apply_along_axis(b_function, 1, pointN) * 1.0
            b[np.where(np.isnan(b))] = 0.0
            b[np.where(np.isinf(b))] = 0.0
            b[np.where(np.abs(b) > 1.0)] /= 255.0
            b[np.where(np.abs(b) < 1.0)] *= 255.0
            if b.shape[1] < 2:
                b = np.hstack((b, np.zeros((b.shape[0], 1))))
            b = b.mean(axis=1)
            b = np.reshape(b, img_size)

            striped = np.stack((r, g, b), axis=-1)
            print((striped.shape))
            images.append(striped)
        return images

    def prepare_inputs(self, point, imgsize):
        # return normalizexy(point, imgsize) + (getcenterdistance(point,
        # imgsize),)
        center = (int(imgsize[0] / 2.0),
                  int(imgsize[1] / 2.0))
        return point + (sqrt(pow(point[0] - center[0], 2)
                             + pow(point[1] - center[1], 2)),)

    def getXY(self, i):
        return ((i % self.gridsize) * self.imgsize[0], int(float(i) / self.gridsize) * self.imgsize[1])

    def getIndex(self, pos):
        return int(int(pos[0] / self.imgsize[0]) + int(pos[1] / self.imgsize[1] * self.gridsize)) - 1

    def blit_images(self, surfaces, images):
        for i in range(len(images)):
            blit_array(surfaces[i], images[i])
        pygame.display.flip()

    def show_images(self, images, numimgs=4):
        self._running = True
        size = weight, height = (int(sqrt(numimgs)) * self.imgsize[0], int(sqrt(numimgs)) * self.imgsize[1])
        self.gridsize = sqrt(numimgs)
        striped = zeros((self.imgsize[0], self.imgsize[1], 3))
        pygame.init()
        pygame.key.set_repeat()
        pygame.display.set_caption('''Mouse:: Right - dump circuit (temp.png) :: Left - select :: space to continue''')
        self.display_surf = pygame.display.set_mode(size)
        self.surfaces = []
        selected = []
        pop = []
        pause = False
        self.zoompanel = pygame.sprite.Sprite()
        self.zoompanel.rect = pygame.Rect(0, 0, 2 * size[0], 2 * size[1])
        running = True
        for i in range(len(images)):
            self.surfaces.append(self.display_surf.subsurface(self.getXY(i) + self.imgsize))
        self.blit_images(self.surfaces, images)

    def on_event(self, event, images, pop):
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.selected.append(self.getIndex(event.pos))
                print(self.selected)
            if event.button == 3:
                print(pop[self.getIndex(event.pos)]['phenotype'])
            else:
                print(event)
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                self.pause = True
            elif event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.key == pygame.K_z:
                self.blit_images(self.surfaces, images)
                pygame.display.update()
            else:
                print(event)
                # pause app and get new pop

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z:
                pos = pygame.mouse.get_pos()
                self.zoomsurface(self.surfaces[self.getIndex(pos)], self.zoompanel)
                self.display_surf.blit(self.zoompanel.image, self.zoompanel.rect)
                pygame.display.update()

        if event.type == pygame.QUIT:
            self._running = False

    def evaluate(self, population):
        self._running = False
        self.pause = False
        self.selected = []
        images = self.render_images(population, self.imgsize)
        print("done")
        self.show_images(images, len(population))
        while self._running and not self.pause:
            for event in pygame.event.get():
                self.on_event(event, images, population)
        for i in range(len(population)):
            if i in self.selected:
                if population[i]['fitness'] is None:
                    population[i]['fitness'] = -1 * self.selected.count(i)
                else:
                    population[i]['fitness'] = population[i]['fitness'] - self.selected.count(i)
            else:
                if population[i]['fitness'] is None:
                    population[i]['fitness'] = 0
                else:
                    population[i]['fitness'] = population[i]['fitness'] + 1
        return population



if __name__ == "__main__":
    import sge
    eval_func = Artist()
    sge.evolutionary_algorithm(evaluation_function=eval_func)