<h1 align="left">Precise Localization for Anatomo-Physiological Hallmarks of the Cervical Spine by Using Neural Memory Ordinary Differential Equation</h1> 

## Project Overview
This project aims to achieve precise localization of anatomical and physiological hallmarks of the cervical spine using a Neural Memory Ordinary Differential Equation (nmODE). 

## Installation 
Download and install Miniconda from the https://docs.anaconda.com/free/miniconda/

Create and activate a new Conda environment.
```bash
conda create -n nm_ode_env python=3.8
conda activate nm_ode_env
```

We use PyTorch 2.0.1, and mmcv 2.0.0 for the experiments.
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv=2.0.0"
```

Install torchdiffeq, it is a library in PyTorch used for solving ordinary differential equations (ODEs) and partial differential equations (PDEs).
```bash
pip install torchdiffeq
```

## Usage
Organize your dataset in the following structure.
```txt
├── data
     ├── train images
     ├── val images
     └── annotations: 
     		  ├── keypoints_train.json
     		  └── keypoints_val.json
```

Training and testing.
```bash
python setyp.py install
# train
python tools\train.py <your config path>
```

```bash
# test
python tools\test.py <your config path>
```

Example.
```bash
python tools\train.py .\configs\body_2d_keypoint\nf_dekr\coco\NF-DEKR_hrnetw32.py
```

Neural Memory Ordinary Differential Equation (nmODE)
```bash
class nmodeblock(nn.Module):
    def __init__(self, input, output, eval_times=(0, 1)):
        super(nmodeblock, self).__init__()
        self.input = input
        self.output = output
        self.nmODE_down = nmODE()
        self.ode_down = ODEBlock(self.nmODE_down)
        self.eval_times = torch.tensor(eval_times).float().cuda()

    def forward(self, x):
        # ode
        self.nmODE_down.fresh(x)
        x = self.ode_down(torch.zeros_like(x), self.eval_times)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc, tol=1e-3, adjoint=False):
        """
        Code adapted from https://github.com/EmilienDupont/augmented-neural-odes

        Utility class that wraps odeint and odeint_adjoint.

        Args:
            odefunc (nn.Module): the module to be evaluated
            tol (float): tolerance for the ODE solver
            adjoint (bool): whether to use the adjoint method for gradient calculation
        """
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float()
        else:
            integration_time = eval_times.type_as(x)

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            out = odeint(self.odefunc, x, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out[1]

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)


class nmODE(nn.Module):
    def __init__(self):
        """
        """
        super(nmODE, self).__init__()
        self.nfe = 0  # Number of function evaluations
        self.gamma = None
        self.relu = nn.ReLU(inplace=True)

    def fresh(self, gamma):
        self.gamma = gamma

    def forward(self, t, p):
        self.nfe = self.nfe + 1
        dpdt = -p + torch.pow(torch.sin(p + self.gamma), 2)
        return dpdt
```

## Acknowledge
We acknowledge the excellent implementation from [mmpose](https://github.com/open-mmlab/mmpose).
