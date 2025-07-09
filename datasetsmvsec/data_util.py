import numpy as np
import torch


def random_shift_events(events, max_shift=20, resolution=(180, 240), bounding_box=None):
    """Randomly shift events and crops """
    H, W = resolution
    if bounding_box is not None:
        x_shift = np.random.randint(-min(bounding_box[0, 0], max_shift),
                                    min(W - bounding_box[2, 0], max_shift), size=(1,))
        y_shift = np.random.randint(-min(bounding_box[0, 1], max_shift),
                                    min(H - bounding_box[2, 1], max_shift), size=(1,))
        bounding_box[:, 0] += x_shift
        bounding_box[:, 1] += y_shift
    else:
        x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))

    events[:, 0] += x_shift
    events[:, 1] += y_shift

    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]

    if bounding_box is None:
        return events

    return events, bounding_box


def random_flip_events_along_x(events, resolution=(180, 240), p=0.5, bounding_box=None):
    H, W = resolution
    flipped = False
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]
        flipped = True

    if bounding_box is None:
        return events

    if flipped:
        bounding_box[:, 0] = W - 1 - bounding_box[:, 0]
        bounding_box = bounding_box[[1, 0, 3, 2]]

    return events, bounding_box


def generate_input_representation(events, event_representation, shape, nr_temporal_bins=7):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    if event_representation == 'histogram':
        return generate_event_histogram(events, shape)
    elif event_representation == 'voxel_grid':
        return generate_voxel_grid(events, shape, nr_temporal_bins)


def generate_event_histogram(events, shape):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    height, width = shape
    x, y, t, p = events.T
    x = x.astype(np.int)
    y = y.astype(np.int)
    img_pos = np.zeros((height * width,), dtype="float32")
    img_neg = np.zeros((height * width,), dtype="float32")

    np.add.at(img_pos, x[p == 1] + width * y[p == 1], 1)
    np.add.at(img_neg, x[p == -1] + width * y[p == -1], 1)

    histogram = np.stack([img_neg, img_pos], 0).reshape((2, height, width))

    return histogram


def normalize_event_tensor(event_tensor):
    """Normalize the sensor according the 98 quantile"""
    event_volume_flat = event_tensor.flatten()
    nonzero = np.nonzero(event_volume_flat)
    nonzero_values = event_volume_flat[nonzero]
    if nonzero_values.shape[0]:
        # lower = np.percentile(nonzero_values, 2, interpolation='nearest')
        # upper = np.percentile(nonzero_values, 98, interpolation='nearest')
        # max_val = max(abs(lower), upper)
        max_val = np.percentile(nonzero_values, 98, interpolation='nearest')
        event_tensor = np.clip(event_tensor, 0, max_val)
        event_tensor /= max_val

    return event_tensor


def generate_voxel_grid(events, shape, nr_temporal_bins):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param nr_temporal_bins: number of bins in the temporal axis of the voxel grid
    :param shape: dimensions of the voxel grid
    """
    height, width = shape
    num_bins = nr_temporal_bins
    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 2] = (num_bins - 1) * (events[:, 2] - first_stamp) / deltaT
    ts = events[:, 2]
    xs = events[:, 0].astype(np.int)
    ys = events[:, 1].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


def random_crop_resize(tensor, mid_point, crop_range=[-10, 10], scale_range=[0.8, 1]):
    """Randomly crops a tensor based on the specified mid_point and height and scale range"""
    _, height, width = tensor.shape
    random_delta = torch.rand([2], device='cpu').numpy() * (crop_range[1] - crop_range[0]) + crop_range[0]
    random_scale = torch.rand([2], device='cpu').numpy() * (scale_range[1] - scale_range[0]) + scale_range[0]

    random_delta = np.minimum(random_delta, mid_point) * (random_delta >= 0) + \
                   np.maximum(random_delta, mid_point - np.array([height, width])) * (random_delta < 0)

    left_corner_u = int(np.maximum(0, random_delta[0]))
    left_corner_v = int(np.maximum(0, random_delta[1]))
    right_corner_u = int(np.minimum(height, random_delta[0] + height * random_scale[0]))
    right_corner_v = int(np.minimum(width, random_delta[1] + width * random_scale[1]))

    tensor = tensor[:, left_corner_u:right_corner_u, left_corner_v:right_corner_v]
    tensor = torch.nn.functional.interpolate(tensor[None, :, :, :], (height, width)).squeeze(axis=0)

    return tensor
