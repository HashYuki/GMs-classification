import numpy as np

__all__ = ["Compose", "Gaus_noise", "Shear", "RandomFlip"]


"""
support data shape: (C T V) or (N C T V)
"""


class Compose:
    """
    Reference:
    https://github.com/pytorch/vision/blob/0dceac025615a1c2df6ec1675d8f9d7757432a49/torchvision/transforms/transforms.py#L60
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        return input


class Gaus_noise:
    """
    Reference:
    https://github.com/LZU-SIAT/AS-CAL/blob/966328ae65bb16ba9b7aab153d8150c08c26c81f/feeders/tools.py#L279
    """

    def __init__(self, mean: float = 0, std: float = 0.005, p: float = 0.1):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, input):
        if np.random.random() < self.p:
            noise = np.random.normal(self.mean, self.std, size=input.shape)
            input = input + noise
        return input


class Shear:
    """A linear mapping matrix that displaces each joint in a fixed direction
    Reference:
    https://github.com/LZU-SIAT/AS-CAL/blob/966328ae65bb16ba9b7aab153d8150c08c26c81f/feeders/tools.py#L296
    """

    def __init__(self, r: float = 0.25):
        """
        Args:
            r (float): range of uniform random numbers
            center (int): center of body
        """
        # args
        self.s1_list = [np.random.uniform(-r, r), np.random.uniform(-r, r)]

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Args:
            input (numpy.ndarray): skeleton sequence.

        Returns:
            np.ndarray: transdformed skeleton sequence.
        """
        R = np.array(
            [
                [1, self.s1_list[0]],
                [
                    self.s1_list[1],
                    1,
                ],
            ]
        )
        if input.ndim == 3:
            input = np.einsum("C T V, I C -> I T V", input, R)
        elif input.ndim == 4:
            input = np.einsum("N C T V, I C -> N I T V", input, R)

        return input


class RandomFlip:
    """Interpolate"""

    def __init__(self, flip_pair, p=0.5):
        """
        Args:
            p (float): interpolate_ratio
        """
        # args
        self.flip_pair = np.array(flip_pair)
        self.p = p

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Args:
            input (numpy.ndarray): skeleton sequence.

        Returns:
            np.ndarray: transdformed skeleton sequence.
        """
        if np.random.random() < self.p:
            if input.ndim == 3:
                input[0] *= -1
                input[:, :, self.flip_pair[:, 0]], input[:, :, self.flip_pair[:, 1]] = (
                    input[:, :, self.flip_pair[:, 1]],
                    input[:, :, self.flip_pair[:, 0]],
                )
            elif input.ndim == 4:
                input[:, 0] *= -1
                (
                    input[:, :, :, self.flip_pair[:, 0]],
                    input[:, :, :, self.flip_pair[:, 1]],
                ) = (
                    input[:, :, :, self.flip_pair[:, 1]],
                    input[:, :, :, self.flip_pair[:, 0]],
                )
        return input


class AddBoneChannel:
    """Add Bone Channel"""

    def __init__(self, inward):
        """
        Args:
        """
        # args
        self.inward = np.array(inward)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Args:
            input (numpy.ndarray): skeleton sequence.

        Returns:
            np.ndarray: transdformed skeleton sequence.
        """
        bone = np.zeros_like(input)
        for v1, v2 in self.inward:
            if input.ndim == 3:
                bone[:, :, v1] = input[:, :, v1] - input[:, :, v2]
                new_data = np.concatenate([input, bone], axis=0)
            elif input.ndim == 4:
                bone[:, :, :, v1] = input[:, :, :, v1] - input[:, :, :, v2]
                new_data = np.concatenate([input, bone], axis=1)
        return new_data


class AddVelAcc:
    """Add Vel/Acc Channels"""

    def __init__(self, add_vel=True, add_acc=True):
        """
        Args:
        """
        # args
        self.add_vel = add_vel
        self.add_acc = add_acc

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Args:
            input (numpy.ndarray): skeleton sequence.

        Returns:
            np.ndarray: transdformed skeleton sequence.
        """
        # velocity
        vel = np.zeros_like(input)
        vel[:, 2:-2] = input[:, 4:] + input[:, 3:-1] - input[:, 1:-3] - input[:, :-4]

        # acc
        if self.add_acc:
            acc = np.zeros_like(input)
            acc[:, 2:-2] = vel[:, 4:] + vel[:, 3:-1] - vel[:, 1:-3] - vel[:, :-4]

        if self.add_vel:
            input = np.concatenate([input, vel], axis=0)
        if self.add_acc:
            input = np.concatenate([input, acc], axis=0)
        return input


class OnlyVelAcc:
    """Add Vel/Acc Channels"""

    def __init__(self, add_vel=True, add_acc=True):
        """
        Args:
        """
        # args
        self.add_vel = add_vel
        self.add_acc = add_acc

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Args:
            input (numpy.ndarray): skeleton sequence.

        Returns:
            np.ndarray: transdformed skeleton sequence.
        """
        # velocity
        vel = np.zeros_like(input)
        vel[:, 2:-2] = input[:, 4:] + input[:, 3:-1] - input[:, 1:-3] - input[:, :-4]

        # acc
        if self.add_acc:
            acc = np.zeros_like(input)
            acc[:, 2:-2] = vel[:, 4:] + vel[:, 3:-1] - vel[:, 1:-3] - vel[:, :-4]

        if self.add_vel:
            return vel
        if self.add_acc:
            return acc
