# Comparison: decompose_homography_my vs decompose_homography_cursor

## Key Differences

### 1. **Mathematical Approach**

**decompose_homography_my:**
- Direct algebraic manipulation
- Uses the formula: `H[:2, :2] = sRK + tv^T`
- Extracts: `sRK = H[:2, :2] - t@v.T`
- Scale: `s = sqrt(det(sRK))`

**decompose_homography_cursor:**
- Step-by-step removal approach
- First removes perspective: `H_affine = H @ Hp^-1`
- Then decomposes affine part using QR
- Scale extracted via SVD of Q_scaled

### 2. **K Matrix Normalization**

**decompose_homography_my:**
- ❌ Does NOT normalize K to have det(K) = 1
- K comes directly from QR decomposition of RK/s
- Only ensures positive diagonal via sign correction

**decompose_homography_cursor:**
- ✅ Normalizes K so that det(K) = 1
- Uses: `K = R / sqrt(det(R))` where R is from QR
- This ensures K has unit determinant

### 3. **Hp Construction**

**decompose_homography_my:**
- Sets `Hp[2, :2] = v` (perspective vector)
- Leaves `Hp[2, 2] = 1` (from np.eye(3))
- Does not use `H[2, 2]` value

**decompose_homography_cursor:**
- Sets `Hp[2, :2] = v` (perspective vector)
- Sets `Hp[2, 2] = w` where `w = H[2, 2]`
- Properly preserves the full perspective component

### 4. **Scale and Rotation Extraction**

**decompose_homography_my:**
- Scale: `s = sqrt(det(sRK))`
- Rotation: Directly from QR decomposition `R, K = qr(RK/s)`
- Sign correction applied to ensure positive diagonal

**decompose_homography_cursor:**
- Scale: `s = sqrt(S[0] * S[1])` (geometric mean of SVD singular values)
- Rotation: Extracted via SVD: `R_rot = U @ Vt`
- More robust extraction method

### 5. **Translation Handling**

**decompose_homography_my:**
- Translation `t` goes directly to `Hs[:2, 2]`
- `Ha[:2, 2]` is implicitly 0 (from np.eye(3))

**decompose_homography_cursor:**
- Translation from `H_affine[:2, 2]` goes to `Hs[:2, 2]`
- Explicitly sets `Ha[:2, 2] = 0.0` (no translation in Ha)

### 6. **Error Handling**

**decompose_homography_my:**
- ❌ No error handling
- ❌ No verification
- Will crash on invalid input

**decompose_homography_cursor:**
- ✅ Try-except block
- ✅ Verification code that checks reconstruction
- ✅ Prints warnings if decomposition fails
- ✅ Returns identity matrices on error

### 7. **Input Normalization**

**decompose_homography_my:**
- Assumes H is already normalized
- Comment says: "already normalized before calling this function"

**decompose_homography_cursor:**
- Normalizes H internally: `H = H / H[2, 2]`
- More robust to unnormalized input

## Potential Issues in decompose_homography_my

1. **det(K) ≠ 1**: The requirement that Ha should have det(K) = 1 is not satisfied
2. **Hp[2, 2] not set correctly**: Should be `H[2, 2]` not 1
3. **No error handling**: Will crash on edge cases
4. **Scale calculation**: Using `sqrt(det(sRK))` may not be correct if det(sRK) < 0

## Recommendation

The `decompose_homography_cursor` function is more robust and follows the correct mathematical formulation:
- ✅ Ensures det(K) = 1
- ✅ Properly handles Hp[2, 2]
- ✅ Has error handling
- ✅ Includes verification

The `decompose_homography_my` function has a simpler approach but may not satisfy all constraints.
