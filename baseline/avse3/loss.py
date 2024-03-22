import keras.ops as ops


def l2_norm(s1, s2):
    norm = ops.sum(s1 * s2, -1, keepdims=True)
    return norm


def si_snr_loss(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = ops.convert_to_tensor(s1) - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * ops.log10((target_norm) / (noise_norm + eps) + eps)
    return -1 * ops.mean(snr)