# Why decompose_homography_my and decompose_homography_cursor produce different Hs and Ha

## The Key Difference: Scale Distribution

The two methods distribute scale differently between Hs and Ha, which leads to different results.

## decompose_homography_my Approach

1. **Extracts scale from sRK:**
   ```python
   sRK = H[:2, :2] - t@v.T
   s = sqrt(det(sRK))  # Scale extracted here
   RK = sRK/s  # Normalized so det(RK) = 1
   ```

2. **QR decomposition:**
   ```python
   R, K = qr(RK)  # det(RK) = det(R) * det(K) = 1
   ```
   Since R is orthogonal (det(R) = ±1), we get det(K) = ±1
   But K is NOT explicitly normalized to det(K) = 1

3. **Scale goes entirely to Hs:**
   ```python
   Hs[:2, :2] = s * R  # All scale in Hs
   Ha[:2, :2] = K      # K has det(K) ≈ 1 (not exactly normalized)
   ```

**Result:** 
- Hs has scale `s = sqrt(det(sRK))`
- Ha has K with det(K) ≈ 1 (but not exactly 1)

## decompose_homography_cursor Approach

1. **QR decomposition first:**
   ```python
   Q, R = qr(H_affine[:2, :2])  # det(R) can be any positive value
   ```

2. **Normalize K to have det(K) = 1:**
   ```python
   det_R = det(R)
   scale_R = sqrt(det_R)  # Extract scale from R
   K = R / scale_R        # K now has det(K) = 1 exactly
   Q_scaled = Q * scale_R # Scale moved to Q_scaled
   ```

3. **Extract scale from Q_scaled:**
   ```python
   U, S, Vt = svd(Q_scaled)
   s = sqrt(S[0] * S[1])  # Geometric mean
   R_rot = U @ Vt
   ```

4. **Scale goes to Hs:**
   ```python
   Hs[:2, :2] = s * R_rot  # Scale in Hs
   Ha[:2, :2] = K          # K has det(K) = 1 exactly
   ```

**Result:**
- Hs has scale `s` extracted from Q_scaled via SVD
- Ha has K with det(K) = 1 exactly (explicitly normalized)

## Why They're Different

The fundamental difference is:

1. **decompose_homography_my:**
   - Scale: `s = sqrt(det(sRK))`
   - K: det(K) ≈ 1 (from QR, but not explicitly normalized)
   - **Scale distribution:** All scale in Hs, K keeps its natural determinant

2. **decompose_homography_cursor:**
   - Scale: `s` extracted via SVD after normalizing K
   - K: det(K) = 1 exactly (explicitly normalized)
   - **Scale distribution:** Scale redistributed to ensure det(K) = 1

## Mathematical Relationship

Both decompositions are valid, but they satisfy different constraints:

- **decompose_homography_my:** `H = Hs * Ha * Hp` where det(K) ≈ 1
- **decompose_homography_cursor:** `H = Hs * Ha * Hp` where det(K) = 1 exactly

The scale difference comes from the fact that:
- In `decompose_homography_my`: `s_my = sqrt(det(sRK))`
- In `decompose_homography_cursor`: `s_cursor` is extracted after normalizing K, so it absorbs the scale that was in K

## Example from Your Output

**decompose_homography_my:**
- Hs scale: ~1.018 (from sqrt(0.98888104² + 0.24449027²))
- Ha det(K): 1.62496814 * 0.61539668 ≈ 1.000 (approximately 1)

**decompose_homography_cursor:**
- Hs scale: ~1.096 (from sqrt(1.07358956² + 0.22249276²))
- Ha det(K): 1.52122573 * 0.65736464 = 1.000 (exactly 1)

The cursor version has a larger scale in Hs because it absorbed the scale that was redistributed from K to ensure det(K) = 1.

## Which is Correct?

Both are mathematically valid decompositions, but:
- **decompose_homography_cursor** satisfies the constraint **det(K) = 1** exactly
- **decompose_homography_my** has det(K) ≈ 1 but not exactly 1

If the requirement is that Ha should have det(K) = 1, then `decompose_homography_cursor` is the correct implementation.
