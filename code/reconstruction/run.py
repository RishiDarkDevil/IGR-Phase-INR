import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from pyhocon import ConfigFactory
import numpy as np
import argparse
import GPUtil
import torch
import utils.general as utils
from model.sample import Sampler
from model.network import gradient
from scipy.spatial import cKDTree
from utils.plots import plot_surface, plot_cuts
import open3d


class ReconstructionRunner:

    def run(self):

        print("running")

        self.data = self.data.cuda()
        self.data.requires_grad_()

        if self.eval:

            print("evaluating epoch: {0}".format(self.startepoch))
            my_path = os.path.join(self.cur_exp_dir, 'evaluation', str(self.startepoch))

            utils.mkdir_ifnotexists(os.path.join(self.cur_exp_dir, 'evaluation'))
            utils.mkdir_ifnotexists(my_path)
            self.plot_shapes(epoch=self.startepoch, path=my_path, with_cuts=True)
            return

        print("training")

        # PHASE IMPLEMENTATION: START
        for epoch in range(self.startepoch, self.nepochs + 1):

            # 1. Pick a random set of points from the point cloud dataset of batch_size `self.points_batch` to train
            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))

            # 2. Get the points for the above indices
            cur_data = self.data[indices] # shape: points_batch, dimension of space = 3 or 2

            # 3. Sample points from the balls in the current choice of randomly chosen points from the point cloud
            cur_ball_pts = torch.tensor([
               utils.sample_ball(point, self.conf.get_float('train.ball_sigma'), self.conf.get_int('train.pts_per_ball')) 
               for point in cur_data
              ]).cuda() # shape: points_batch, n_points in each ball, dimension of space = 3

            # For making prediction using NN for INR model, we flatten the first axis which we revert later
            ball_pts = cur_ball_pts.view(-1, cur_ball_pts.shape[-1]) # shape: points_batch * n_points in each ball, dimension of space = 3

            # 4. Sample points from omega
            omega_pts = utils.sample_omega(self.omega_coords, self.conf.get_int('train.pts_in_omega')).cuda() # shape: n_points in omega, dimension of space = 3

            # Sae checkpoints and plot (Same as that in IGR)
            if epoch % self.conf.get_int('train.checkpoint_frequency') == 0:
                print('saving checkpoint: ', epoch)
                self.save_checkpoints(epoch)
                print('plot validation epoch: ', epoch)
                self.plot_shapes(epoch)

            # change back to train mode (Same as that in IGR)
            self.network.train()
            self.adjust_learning_rate(epoch)

            # 5. Estimate Reconstruction Loss

            reconstruction_pred = self.network(ball_pts) # shape: (points_batch * pts_per_ball, 1)
            # We need to divide the function value at the sampled points by the pdf using which we sampled points in the ball
            # In our case Normal, This is done for Monte-Carlo Estimation
            reconstruction_pred_normal = torch.tensor(
              [utils.normal_pdf(cur_data[idx][None,:], self.conf.get_float('train.ball_sigma'), pts) for idx, pts in enumerate(cur_ball_pts)]
              ).cuda() # shape: (points_batch, pts_per_ball, 1)
            reconstruction_pred = reconstruction_pred.view(cur_ball_pts.shape[0], cur_ball_pts.shape[1], -1) # shape: (points_batch, pts_per_ball, 1)
            monte_carlo_estimand = reconstruction_pred / reconstruction_pred_normal # shape: (points_batch, pts_per_ball, 1)
            # Monte-Carlo Estimation of the Integral for Reconstruction Loss
            reconstruction_loss = (monte_carlo_estimand.sum(axis=1) / cur_ball_pts.shape[1]).abs().mean() 

            # 6. Estimate Waals-Cahn-Hilliard (WCH) Loss

            WCH_pred = self.network(omega_pts) # shape: (n_points in omega, 1)
            grad = gradient(omega_pts, WCH_pred)
            W_u = WCH_pred ** 2 - 2 * torch.abs(WCH_pred) + 1
            # Monte-Carlo Estimation of the Integral for WCH Loss
            # Here we divide by the Uniform pdf which is same as simply multiplying by omega's volume
            WCH_loss = self.omega_vol * (self.epsilon * grad.norm(2, dim=-1) ** 2 + W_u).mean(dim=0)

            # The final loss
            loss = self.lmbda * reconstruction_loss + WCH_loss

            # 7. Estimate Additional Normal Loss

            if self.mu > 0.0:
                u_x = self.network(cur_data[:, :self.d_in]).view(-1, 1) # shape: points_batch, 1
                grad_w_x = (self.epsilon ** 0.5) * u_x # shape: points_batch, 1
                if self.has_normals:
                    normals = cur_data[:, -self.d_in:] # shape: points_batch, dimension = 3 or 2
                    normals_loss = (normals - grad_w_x).norm(1, dim=1).mean()
                else:
                    normals_loss = (1.0 - grad_w_x.norm(2, dim=1)).view(-1, 1).norm(2, dim=1).mean()

                # The final loss term is the sum of all the above three losses
                loss += self.mu * normals_loss

            # back propagation

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            if epoch % self.conf.get_int('train.status_frequency') == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tReconstruction loss: {:.6f}'
                    '\tWCH loss: {:.6f}\tNormals Loss: {:.6f}'.format(
                    epoch, self.nepochs, 100. * epoch / self.nepochs,
                    loss.item(), reconstruction_loss.item(), WCH_loss.item(), normals_loss.item()))

        # PHASE IMPLEMENTATION: END

    def plot_shapes(self, epoch, path=None, with_cuts=False):
        # plot network validation shapes
        with torch.no_grad():

            self.network.eval()

            if not path:
                path = self.plots_dir

            indices = torch.tensor(np.random.choice(self.data.shape[0], self.points_batch, False))

            pnts = self.data[indices, :3]

            plot_surface(with_points=True,
                         points=pnts,
                         decoder=self.network,
                         path=path,
                         epoch=epoch,
                         shapename=self.expname,
                         **self.conf.get_config('plot'))

            if with_cuts:
                plot_cuts(points=pnts,
                          decoder=self.network,
                          path=path,
                          epoch=epoch,
                          near_zero=False)

    def __init__(self, **kwargs):

        self.home_dir = os.path.abspath(os.pardir)

        # config setting

        if type(kwargs['conf']) == str:
            self.conf_filename = './reconstruction/' + kwargs['conf']
            self.conf = ConfigFactory.parse_file(self.conf_filename)
        else:
            self.conf = kwargs['conf']

        self.expname = kwargs['expname']

        # GPU settings

        self.GPU_INDEX = kwargs['gpu_index']

        if not self.GPU_INDEX == 'ignore':
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        self.num_of_gpus = torch.cuda.device_count()

        self.eval = kwargs['eval']

        # settings for loading an existing experiment

        if (kwargs['is_continue'] or self.eval) and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(self.home_dir, 'exps', self.expname)):
                timestamps = os.listdir(os.path.join(self.home_dir, 'exps', self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue'] or self.eval

        self.exps_folder_name = 'exps'

        utils.mkdir_ifnotexists(utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name)))

        self.input_file = self.conf.get_string('train.input_path')
        self.data = utils.load_point_cloud_by_file_extension(self.input_file)

        # PHASE IMPLEMENTATION: START

        # 1. We normalize the point set (`self.data`) to have unit max norm
        self.data /= max(np.linalg.norm(self.data, axis=1))

        # 2. Omega is taken as the Axis Aligned Bounding Box for the points in the point cloud, scaled 1.5 
        # for 3D and scaled 2 for 2D.
        points = open3d.utility.Vector3dVector(np.array(self.data))
        bounding_box = open3d.geometry.AxisAlignedBoundingBox().create_from_points(points)
        bounding_box = bounding_box.scale(self.conf.get_float('train.bounding_box_scale'), bounding_box.get_center())
        self.omega_coords = np.asarray(bounding_box.get_box_points()) # We need to keep the coords of the bounding box for characterizing omega
        self.omega_vol = bounding_box.volume() # We need the volume of omega

        self.epsilon = self.conf.get_float('network.loss.epsilon')
        self.lmbda = self.conf.get_float('network.loss.lambda')
        self.mu = self.conf.get_float('network.loss.mu')

        self.has_normals = self.conf.get_bool('network.has_normals') and (self.data.shape[-1] >= 6)

        # PHASE IMPLEMENTATION: END

        sigma_set = []
        ptree = cKDTree(self.data)

        for p in np.array_split(self.data, 100, axis=0):
            d = ptree.query(p, 50 + 1)
            sigma_set.append(d[0][:, -1])

        sigmas = np.concatenate(sigma_set)
        self.local_sigma = torch.from_numpy(sigmas).float().cuda()

        self.expdir = utils.concat_home_dir(os.path.join(self.home_dir, self.exps_folder_name, self.expname))
        utils.mkdir_ifnotexists(self.expdir)

        if is_continue:
            self.timestamp = timestamp
        else:
            self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

        self.cur_exp_dir = os.path.join(self.expdir, self.timestamp)
        utils.mkdir_ifnotexists(self.cur_exp_dir)

        self.plots_dir = os.path.join(self.cur_exp_dir, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.checkpoints_path = os.path.join(self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.checkpoints_path = os.path.join(self.cur_exp_dir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))

        self.nepochs = kwargs['nepochs']

        self.points_batch = kwargs['points_batch']

        self.global_sigma = self.conf.get_float('network.sampler.properties.global_sigma')
        self.sampler = Sampler.get_sampler(self.conf.get_string('network.sampler.sampler_type'))(self.global_sigma,
                                                                                                 self.local_sigma)
        self.grad_lambda = self.conf.get_float('network.loss.lambda')
        self.normals_lambda = self.conf.get_float('network.loss.normals_lambda')

        self.d_in = self.conf.get_int('train.d_in')

        self.network = utils.get_class(self.conf.get_string('train.network_class'))(d_in=self.d_in,
                                                                                    **self.conf.get_config(
                                                                                        'network.inputs'))

        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))
        self.weight_decay = self.conf.get_float('train.weight_decay')

        self.startepoch = 0

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0),
                    "weight_decay": self.weight_decay
                },
            ])

        # if continue load checkpoints

        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.network.load_state_dict(saved_model_state["model_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.startepoch = saved_model_state['epoch']

    def get_learning_rate_schedules(self, schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self, epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--points_batch', type=int, default=10000, help='point batch size')
    parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='setup.conf')
    parser.add_argument('--expname', type=str, default='single_shape')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true", help='continue')
    parser.add_argument('--timestamp', default='latest', type=str)
    parser.add_argument('--checkpoint', default='latest', type=str)
    parser.add_argument('--eval', default=False, action="store_true")

    args = parser.parse_args()

    if args.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = args.gpu

    trainrunner = ReconstructionRunner(
            conf=args.conf,
            points_batch=args.points_batch,
            nepochs=args.nepoch,
            expname=args.expname,
            gpu_index=gpu,
            is_continue=args.is_continue,
            timestamp=args.timestamp,
            checkpoint=args.checkpoint,
            eval=args.eval
    )

    trainrunner.run()
