from petsc4py import PETSc
import numpy as np

n = 10

viewer = PETSc.Viewer().createBinary("vec.dat", "r")
vec = PETSc.Vec().load(viewer)
tmp = vec.getArray()
assert np.array_equal(tmp, 0.1 * np.arange(n))

viewer = PETSc.Viewer().createBinary("mat.dat", "r")
A = PETSc.Mat().load(viewer)
(a, b) = A.getSize()
assert a == n and b == n
for i in range(n):
    cols, vals = A.getRow(i)
    if i > 0:
        assert i - 1 in cols
        idx = list(cols).index(i - 1)
        assert vals[idx] == -0.5 * i
    assert i in cols
    idx = list(cols).index(i)
    assert vals[idx] == i
    if i < n - 1:
        assert i + 1 in cols
        idx = list(cols).index(i + 1)
        assert vals[idx] == 1.5 * i

viewer = PETSc.Viewer().createBinary("vec_block.dat", "r")
vec = PETSc.Vec().load(viewer)
tmp = vec.getArray()
assert np.array_equal(tmp[::2], 0.1 * np.arange(n))
assert np.array_equal(tmp[1::2], 0.2 * np.arange(n))

viewer = PETSc.Viewer().createBinary("mat_block.dat", "r")
A = PETSc.Mat().load(viewer)
block_size = A.getBlockSize()
assert block_size == 2
(a, b) = A.getSize()
assert a == block_size * n and b == block_size * n

a0 = np.array([[0.1, 0.2], [0.3, 0.4]])
a1 = a0 + 1
a2 = a1 + 1

for i in range(n):
    for k in range(block_size):
        irow = block_size * i + k
        cols, vals = A.getRow(irow)
        if i > 0:
            j = i - 1
            for l in range(block_size):
                icol = block_size * j + l
                assert icol in cols
                idx = list(cols).index(icol)
                assert vals[idx] == a0[k, l] * i

        j = i
        for l in range(block_size):
            icol = block_size * j + l
            assert icol in cols
            idx = list(cols).index(icol)
            assert vals[idx] == a1[k, l] * i

        if i < n - 1:
            j = i + 1
            for l in range(block_size):
                icol = block_size * j + l
                assert icol in cols
                idx = list(cols).index(icol)
                assert vals[idx] == a2[k, l] * i

#     break

# ptr, indices, values = A.getValuesCSR()
# tmp = np.repeat(2 * np.diff(ref_ptr), 2)
# print(np.array_equal(np.diff(ptr), tmp))
# print(np.diff(ptr))
# print(np.repeat(np.diff(ref_ptr), 2))
# print(indices)
# print(values)
# print(indices.size, ptr.size, values.size)
# print(dir(A.getValuesCSR()))
