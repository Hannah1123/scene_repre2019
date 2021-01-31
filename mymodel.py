import os
import torch
import imageio
import skimage
import numpy as np
import cv2
from glob import glob
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def parse_intrinsics0(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx/width * trgt_sidelength
        cy = cy/height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, world2cam_poses

def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img

def load_rgb(path, sidelength=None):
    img = imageio.imread(path)[:, :, :3]
    img = skimage.img_as_float32(img)

    img = square_crop_img(img)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_AREA)

    img -= 0.5
    img *= 2.
    img = img.transpose(2, 0, 1)
    return img


def load_depth(path, sidelength=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    if sidelength is not None:
        img = cv2.resize(img, (sidelength, sidelength), interpolation=cv2.INTER_NEAREST)

    img *= 1e-4

    if len(img.shape) == 3:
        img = img[:, :, :1]
        img = img.transpose(2, 0, 1)
    else:
        img = img[None, :, :]
    return img


def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()

def load_params(filename):
    lines = open(filename).read().splitlines()

    params = np.array([float(x) for x in lines[0].split()]).astype(np.float32).squeeze()
    return params


class SceneInstanceDataset():
    """This creates a dataset class for a single object instance (such as a
    single car)."""
    def __init__(
        self,
        instance_idx,
        instance_dir,
        specific_observation_idcs=None,  
        img_sidelength=None,
        num_images=-1):
        """
        specific_observation_idcs: 
          For few-shot case: Can pick specific observations only
        """

        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir

        color_dir = os.path.join(instance_dir, "rgb")
        pose_dir = os.path.join(instance_dir, "pose")
        param_dir = os.path.join(instance_dir, "params")

        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.has_params = os.path.isdir(param_dir)
        self.color_paths = sorted(glob_imgs(color_dir))
        self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

        if self.has_params:
            self.param_paths = sorted(glob(os.path.join(param_dir, "*.txt")))
        else:
            self.param_paths = []

        if specific_observation_idcs is not None:
            # 同一个instance 里指定 idx 
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.pose_paths = pick(self.pose_paths, specific_observation_idcs)
            self.param_paths = pick(self.param_paths, specific_observation_idcs)
        elif num_images != -1:
            # uniform sample images: np.linspace(start, stop, num_samples)

            idcs = np.linspace(0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int)
            self.color_paths = pick(self.color_paths, idcs)
            self.pose_paths = pick(self.pose_paths, idcs)
            self.param_paths = pick(self.param_paths, idcs)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx):
        intrinsics, _, _, _ = parse_intrinsics0(os.path.join(self.instance_dir, "intrinsics.txt"),
                                               trgt_sidelength=self.img_sidelength)

        intrinsics = torch.Tensor(intrinsics).float()

        rgb = load_rgb(self.color_paths[idx], sidelength=self.img_sidelength)
        rgb = rgb.reshape(3, -1).transpose(1, 0)

        pose = load_pose(self.pose_paths[idx])

        if self.has_params:
            params = load_params(self.param_paths[idx])
        else:
            params = np.array([0])

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
            "rgb": torch.from_numpy(rgb).float(),
            "pose": torch.from_numpy(pose).float(),
            "uv": uv,
            "param": torch.from_numpy(params).float(),
            "intrinsics": intrinsics
        }
        return sample

    
    


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x
def parse_intrinsics(intrinsics):
    intrinsics = intrinsics.to(device)

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    return fx, fy, cx, cy
def lift(x, y, z, intrinsics, homogeneous=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)
def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates plus depth to  world coordinates.
    '''
    batch_size, _, _ = cam2world.shape

    x_cam = xy[:, :, 0].view(batch_size, -1)
    y_cam = xy[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)  # (batch_size, -1, 4)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(cam2world, pixel_points_cam).permute(0, 2, 1)[:, :, :3]  # (batch_size, -1, 3)

    return world_coords

def get_ray_directions(xy, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates to normalized directions of rays through these pixels.
    '''
    batch_size, num_samples, _ = xy.shape

    z_cam = torch.ones((batch_size, num_samples)).to(device)
    pixel_points = world_from_xy_depth(xy, z_cam, intrinsics=intrinsics, cam2world=cam2world)  # (batch, num_samples, 3)

    cam_pos = cam2world[:, :3, 3]
    ray_dirs = pixel_points - cam_pos[:, None, :]  # (batch, num_samples, 3)
    ray_dirs = F.normalize(ray_dirs, dim=2)
    return ray_dirs

def depth_from_world(world_coords, cam2world):
    batch_size, num_samples, _ = world_coords.shape

    points_hom = torch.cat((world_coords, torch.ones((batch_size, num_samples, 1)).to(device)),
                           dim=2)  # (batch, num_samples, 4)

    # permute for bmm
    points_hom = points_hom.permute(0, 2, 1)

    points_cam = torch.inverse(cam2world).bmm(points_hom)  # (batch, 4, num_samples)
    depth = points_cam[:, 2, :][:, :, None]  # (batch, num_samples, 1)
    return depth

def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)

class Raymarcher(nn.Module):
    def __init__(self,
             num_feature_channels,
             raymarch_steps):
        super().__init__()

        self.n_feature_channels = num_feature_channels
        self.steps = raymarch_steps

        hidden_size = 16
        self.lstm = nn.LSTMCell(input_size=self.n_feature_channels,
                                hidden_size=hidden_size)

        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)

        self.out_layer = nn.Linear(hidden_size, 1)
#        self.counter = 0

    def forward(self,
            cam2world,
            phi,
            uv,
            intrinsics):
        batch_size, num_samples, _ = uv.shape
        #log = list()

        ray_dirs = get_ray_directions(
            uv,
            cam2world=cam2world,
            intrinsics=intrinsics)

        initial_depth = torch.zeros((batch_size, num_samples, 1))\
            .normal_(mean=0.05, std=5e-4)\
            .to(device)
        init_world_coords = \
            world_from_xy_depth(
            uv,
            initial_depth,
            intrinsics=intrinsics,
            cam2world=cam2world)

        world_coords = [init_world_coords]
        depths = [initial_depth]
        states = [None]

        for step in range(self.steps):
            v = phi(world_coords[-1])

            state = self.lstm(v.view(-1, self.n_feature_channels), states[-1])

            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

            signed_distance = self.out_layer(state[0]).view(batch_size, num_samples, 1)
            new_world_coords = world_coords[-1] + ray_dirs * signed_distance

            states.append(state)
            world_coords.append(new_world_coords)

            depth = depth_from_world(world_coords[-1], cam2world)

            if self.training:
                print("Raymarch step %d: Min depth %0.6f, max depth %0.6f" %
                      (step, depths[-1].min().detach().cpu().numpy(), depths[-1].max().detach().cpu().numpy()))

            depths.append(depth)

        log = None
        return world_coords[-1], depths[-1], log

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.net(input)

class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        # first layer, in_features -> hidden
        self.net.append(
            FCLayer(in_features=in_features, 
                out_features=hidden_ch))

        # all following layers hidden->hidden
        for i in range(num_hidden_layers):
            self.net.append(
                FCLayer( 
                    in_features=hidden_ch, 
                    out_features=hidden_ch))

        last_layer_cls = nn.Linear if outermost_linear \
                else FCLayer
        self.net.append(last_layer_cls(
            in_features=hidden_ch, 
            out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self,item):
        return self.net[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(
                m.weight, 
                a=0.0, 
                nonlinearity='relu', 
                mode='fan_in')

    def forward(self, input):
        return self.net(input)

#==================

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def custom_load(model, path, discriminator=None, overwrite_embeddings=False, overwrite_renderer=False, optimizer=None):
    if os.path.isdir(path):
        checkpoint_path = sorted(glob(os.path.join(path, "*.pth")))[-1]
    else:
        checkpoint_path = path

    whole_dict = torch.load(checkpoint_path)

    if overwrite_embeddings:
        del whole_dict['model']['latent_codes.weight']

    if overwrite_renderer:
        keys_to_remove = [key for key in whole_dict['model'].keys() if 'rendering_net' in key]
        for key in keys_to_remove:
            print(key)
            whole_dict['model'].pop(key, None)

    state = model.state_dict()
    state.update(whole_dict['model'])
    model.load_state_dict(state)

    if discriminator:
        discriminator.load_state_dict(whole_dict['discriminator'])

    if optimizer:
        optimizer.load_state_dict(whole_dict['optimizer'])


def custom_save(model, path, discriminator=None, optimizer=None):
    whole_dict = {'model':model.state_dict()}
    if discriminator:
        whole_dict.update({'discriminator':discriminator.state_dict()})
    if optimizer:
        whole_dict.update({'optimizer':optimizer.state_dict()})

    torch.save(whole_dict, path)

#==================

class MySRN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_feature_ch=256
        self.render_layers=5
        self.l2_loss = nn.MSELoss(reduction="mean")


        self.ray_marcher = Raymarcher(num_feature_channels=self.num_feature_ch,
                            raymarch_steps=10)
        self.phi = FCBlock(
                        hidden_ch=self.num_feature_ch,
                        num_hidden_layers=2,
                        in_features=3,
                        out_features=self.num_feature_ch)
        self.pixel_generator = FCBlock(
                        hidden_ch=self.num_feature_ch,
                        num_hidden_layers=self.render_layers - 1,
                        in_features=self.num_feature_ch,
                        out_features=3,
                        outermost_linear=True)

    def get_regularization_loss(self, prediction, ground_truth):
        """Computes regularization loss on final depth map (L_{depth} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: Regularization loss on final depth map.
        直接加sigmoid 不加这个loss
        """
        _, depth = prediction

        neg_penalty = (torch.min(depth, torch.zeros_like(depth)) ** 2)
        return torch.mean(neg_penalty) * 10000

    def get_image_loss(self, pred_imgs, ground_truth):
        """Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        """
        #pred_imgs, _ = prediction
        trgt_imgs = ground_truth

        trgt_imgs = trgt_imgs.to(device)

        loss = self.l2_loss(pred_imgs, trgt_imgs)
        return loss


    def forward(self, input_instance, gnd=None):

        instance_idcs = input_instance["instance_idx"].long().to(device)
        pose = input_instance["pose"].to(device)
        intrinsics = input_instance["intrinsics"].to(device)
        uv = input_instance["uv"].to(device).to(torch.float32)

        end_world_xyz, end_dep, log = self.ray_marcher(
                                cam2world=pose,
                                intrinsics=intrinsics,
                                uv=uv,
                                phi=self.phi)
        v = self.phi(end_world_xyz)

        image_pred = self.pixel_generator(v)

        return image_pred