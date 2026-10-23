# Soften the shadow-catcher shadow of a transparent render.
#   combined.png : render with shadow catcher (object + shadow), straight alpha
#   mask.png     : same view with the ground hidden -> object coverage only
# Over-compositing: premult(combined) = premult(object) since the shadow is black,
# so only the alpha outside the object changes: blur + lighten the shadow alpha.
# usage: python post_shadow.py combined.png mask.png [strength] [blur_px] [falloff]
#   falloff > 1 keeps the peak darkness but shortens the shadow's tail
import sys
import cv2
import numpy as np

comb_p, mask_p = sys.argv[1], sys.argv[2]
strength = float(sys.argv[3]) if len(sys.argv) > 3 else 0.55
blur = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
falloff = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

comb = cv2.imread(comb_p, cv2.IMREAD_UNCHANGED).astype(np.float64) / 65535.0
mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED).astype(np.float64) / 65535.0
A, M = comb[..., 3], mask[..., 3]
premult = comb[..., :3] * A[..., None]

# shadow alpha behind the object, extended under it (object pixels: assume the
# shadow continues smoothly) so the blur doesn't create a halo at the silhouette
S = np.clip((A - M) / np.maximum(1 - M, 1e-4), 0, 1)
S[M > 0.5] = np.nan
fill = cv2.dilate(np.nan_to_num(S, nan=0.0), np.ones((5, 5)), iterations=6)
S = np.where(np.isnan(S), fill, S)
S = cv2.GaussianBlur(S, (0, 0), blur)
peak = max(S.max(), 1e-6)
S = strength * peak * (S / peak) ** falloff

alpha = M + (1 - M) * S
rgb = np.where(alpha[..., None] > 1e-6, premult / np.maximum(alpha[..., None], 1e-6), 0)
out = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], -1)
cv2.imwrite(comb_p, np.round(out * 65535).astype(np.uint16))
