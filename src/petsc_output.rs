#[cfg(feature = "nalgebra")]
use crate::nalgebra::SparseBlockMat;
use crate::{Result, SparseMatF64};
#[cfg(feature = "nalgebra")]
use nalgebra::SVector;
use std::{
    fs::File,
    io::{BufWriter, Write},
};

type PetscInt = i32;

fn write_ints(writer: &mut BufWriter<File>, data: &[PetscInt]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<PetscInt>());
    for x in data {
        bytes.extend(x.to_be_bytes());
    }
    writer.write_all(&bytes)?;
    Ok(())
}

fn write_doubles(writer: &mut BufWriter<File>, data: &[f64]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<f64>());
    for x in data {
        bytes.extend(x.to_be_bytes());
    }
    writer.write_all(&bytes)?;
    Ok(())
}

pub fn write_sparse_mat_petsc(mat: &SparseMatF64, fname: &str) -> Result<()> {
    let file = File::create(fname)?;
    let mut writer = BufWriter::new(file);

    let n_rows = mat.n() as PetscInt;
    let nnz = mat.nnz() as PetscInt;
    let header = [1211216, n_rows, n_rows, nnz];
    write_ints(&mut writer, &header)?;

    let mut lengths = Vec::with_capacity(mat.n());
    let mut cols = Vec::with_capacity(mat.nnz());
    let mut data = Vec::with_capacity(mat.nnz());

    for row in mat.seq_rows() {
        lengths.push(row.len() as PetscInt);
        for (j, v) in row.iter() {
            cols.push(*j as PetscInt);
            data.push(*v);
        }
    }
    write_ints(&mut writer, &lengths)?;
    write_ints(&mut writer, &cols)?;
    write_doubles(&mut writer, &data)?;

    Ok(())
}

pub fn write_vec_petsc(vec: &[f64], fname: &str) -> Result<()> {
    let file = File::create(fname)?;
    let mut writer = BufWriter::new(file);

    let n_rows = vec.len() as PetscInt;
    let header = [1211214, n_rows];
    write_ints(&mut writer, &header)?;

    write_doubles(&mut writer, &vec)?;

    Ok(())
}

#[cfg(feature = "nalgebra")]
pub fn write_sparse_block_mat_petsc<const N: usize>(
    mat: &SparseBlockMat<N>,
    fname: &str,
) -> Result<()> {
    let file = File::create(fname)?;
    let mut writer = BufWriter::new(file);

    let n_rows = (N * mat.n()) as PetscInt;
    let nnz = (N * N * mat.nnz()) as PetscInt;
    let header = [1211216, n_rows, n_rows, nnz];
    write_ints(&mut writer, &header)?;

    let mut lengths = Vec::with_capacity(mat.n() * N);
    let mut cols = Vec::with_capacity(mat.nnz() * N * N);
    let mut data = Vec::with_capacity(mat.nnz() * N * N);

    for row in mat.seq_rows() {
        for i_b in 0..N {
            lengths.push((row.len() * N) as PetscInt);
            for (j, v) in row.iter() {
                for j_b in 0..N {
                    cols.push((N * j + j_b) as PetscInt);
                    data.push(*v.get((i_b, j_b)).unwrap());
                }
            }
        }
    }
    write_ints(&mut writer, &lengths)?;
    write_ints(&mut writer, &cols)?;
    write_doubles(&mut writer, &data)?;

    let fname = &format!("{fname}.info");
    let file = File::create(fname)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "-matload_block_size {N}")?;

    Ok(())
}

#[cfg(feature = "nalgebra")]
pub fn write_block_vec_petsc<const N: usize>(vec: &[SVector<f64, N>], fname: &str) -> Result<()> {
    let file = File::create(fname)?;
    let mut writer = BufWriter::new(file);

    let n_rows = (N * vec.len()) as PetscInt;
    let header = [1211214, n_rows];
    write_ints(&mut writer, &header)?;

    let data = vec.iter().flatten().cloned().collect::<Vec<_>>();
    write_doubles(&mut writer, &data)?;

    let fname = &format!("{fname}.info");
    let file = File::create(fname)?;
    let mut writer = BufWriter::new(file);
    write!(writer, "-vecload_block_size {N}")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{write_sparse_mat_petsc, write_vec_petsc};
    use crate::SparseMatF64;

    #[test]
    fn test_write_petsc() {
        let n = 10;
        let mut ij = Vec::new();
        let mut vals = Vec::new();
        let mut vec = Vec::new();
        for i in 0..n {
            if i > 0 {
                ij.push([i, i - 1]);
                vals.push(-0.5 * i as f64);
            }
            ij.push([i, i]);
            vals.push(i as f64);
            if i < n - 1 {
                ij.push([i, i + 1]);
                vals.push(1.5 * i as f64);
            }
            vec.push(0.1 * i as f64);
        }
        let mat = SparseMatF64::from_ij(n, ij.iter().cloned(), vals.iter().cloned()).unwrap();
        write_vec_petsc(&vec, "vec.dat").unwrap();
        write_sparse_mat_petsc(&mat, "mat.dat").unwrap();
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn test_write_petsc_block() {
        use crate::{nalgebra::SparseBlockMat, petsc_output::write_sparse_block_mat_petsc};

        use super::write_block_vec_petsc;
        use nalgebra::{SMatrix, SVector};

        let n = 10;
        let mut ij = Vec::new();
        let mut vals = Vec::new();
        let mut vec = Vec::new();
        for i in 0..n {
            if i > 0 {
                ij.push([i, i - 1]);
                vals.push(SMatrix::<f64, 2, 2>::new(
                    0.1 * i as f64,
                    0.2 * i as f64,
                    0.3 * i as f64,
                    0.4 * i as f64,
                ));
            }
            ij.push([i, i]);
            vals.push(SMatrix::<f64, 2, 2>::new(
                1.1 * i as f64,
                1.2 * i as f64,
                1.3 * i as f64,
                1.4 * i as f64,
            ));
            if i < n - 1 {
                ij.push([i, i + 1]);
                vals.push(SMatrix::<f64, 2, 2>::new(
                    2.1 * i as f64,
                    2.2 * i as f64,
                    2.3 * i as f64,
                    2.4 * i as f64,
                ));
            }
            vec.push(SVector::<f64, 2>::new(0.1 * i as f64, 0.2 * i as f64));
        }
        let mat = SparseBlockMat::from_ij(n, ij.iter().cloned(), vals.iter().cloned()).unwrap();
        write_block_vec_petsc(&vec, "vec_block.dat").unwrap();
        write_sparse_block_mat_petsc(&mat, "mat_block.dat").unwrap();
    }
}
