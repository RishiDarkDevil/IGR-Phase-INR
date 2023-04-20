import os

import numpy as np
import torch
from scipy.spatial import cKDTree
import trimesh
import math


def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj


def load_point_cloud_by_file_extension(file_name):

    ext = file_name.split('.')[-1]

    if ext == "npz" or ext == "npy":
        point_set = torch.tensor(np.load(file_name)).float()
    else:
        point_set = torch.tensor(trimesh.load(file_name, ext).vertices).float()

    return point_set

# PHASE IMPLEMENTATION: START

def normal_pdf(mu, sigma, points):
    """
    Evaluates the normal pdf with mean `mu` and std dev `sigma` at each point
    points is a tensor of shape: n_points x dimension
    mu is a tensor of shape: 1 x dimension
    """
    d = points.shape[-1]
    eval_normal = torch.exp(-torch.sum((points - mu)**2, axis=-1) / (2*(sigma**2)))/((math.sqrt(2*math.pi) * sigma)**d)
    return eval_normal[:, None]

def sample_ball(center, sigma, n_points):
    """
    Samples `n_points` balls from a normal dist with mean as `center`
    and std dev `sigma` from a normal distribution.
    """
    points = torch.normal(center.repeat(n_points, 1), sigma)
    return points # shape: n_points, dimension of center = 3 or 2

def sample_omega(box_coords, n_points):
    """
    Samples `n_points` from the bounding box which we take as our omega
    We draw uniform sample from this omega
    """

    min_x, max_x = np.min(np.squeeze(box_coords[:, 0])), np.max(np.squeeze(box_coords[:, 0]))
    min_y, max_y = np.min(np.squeeze(box_coords[:, 1])), np.max(np.squeeze(box_coords[:, 1]))
    min_z, max_z = np.min(np.squeeze(box_coords[:, 2])), np.max(np.squeeze(box_coords[:, 2]))

    sample_points = np.asarray([np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y),
                            np.random.uniform(min_z, max_z)]) for _ in range(n_points)])

    return torch.tensor(sample_points) # shape: n_points, dimension of omega space = 3 or 2

def chamfer_dist(mesh_x, mesh_y, sample_count=10000000):
    """
    Calculates the double-sided Chamfer Distance between `mesh_x` and `mesh_y`
    mesh_x: trimesh.Trimesh
    mesh_y: trimesh.Trimesh
    """
    print('Converting Meshes to Point Clouds...')
    # sample both the meshes densely
    points_x = np.array(trimesh.sample.sample_surface_even(mesh_x, sample_count)[0])
    points_y = np.array(trimesh.sample.sample_surface_even(mesh_y, sample_count)[0])

    print('Creating Search Tree...')
    # Index the points for nearest neighbour retrieval
    points_x_tree = cKDTree(points_x)
    points_y_tree = cKDTree(points_y)

    print('Calculating one-sided Chamfer Distance from 1st to 2nd mesh...')
    # X->Y Chamfer Distance
    chamfer_x_y = points_y_tree.query(points_x, k=1)[0]
    chamfer_x_y = np.mean(chamfer_x_y)

    print('Calculating one-sided Chamfer Distance from 2nd to 1st mesh...')
    # Y->X Chamfer Distance
    chamfer_y_x = points_y_tree.query(points_y, k=1)[0]
    chamfer_y_x = np.mean(chamfer_y_x)

    print('Calculating double-sided Chamfer Distance')
    # Double-Sided Chamfer Distance
    chamfer_dist = (chamfer_x_y + chamfer_y_x) / 2

    return chamfer_dist


# PHASE IMPLEMENTATION: END

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)