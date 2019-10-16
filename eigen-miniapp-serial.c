/* Eigenrockets mini-app for full-to-banded transformation: serial implementation

   USAGE: <executable> <mode> <block size> <matrix size>
   <mode> : 0 for Householder QR algorithm
            1 for Cholesky QR algorithm (not implemented yet)
            2 for modified Cholesky QR algorithm (not implemented yet)
   <block size> : the size of most matrix blocks in level-3 BLAS operations
                  & the bandwidth of transformed matrix
   <matrix size> : the size of the matrix being diagonalized

   NOTES: - for simplicity & rapidity of development, this is written in ANSI C (C89)
            and uses the standard Fortran interfaces to BLAS & LAPACK
          - this implementation is restricted to real doubles
          - "ELPA thesis" refers to the 2012 PhD thesis of Thomas Auckenthaler,
            "Highly Scalable Eigensolvers for Petaflop Applications"
          - only the lower triangle of the symmetric matrix is stored and transformed
            & the Householder vectors are stored in a separate matrix whereas in ELPA,
            the symmetric matrix is fully stored & overlaid w/ Householder vectors
          - block Householder transformations are stored as W & Y in I - W*Y^T
          - a complete set of block Householder transformations are stored in the pattern
            [ W_n W_(n-1) ... W_2     W_1 ]
            [ Y_1 Y_2     ... Y_(n-1) Y_n ]
            contained within the memory footprint of the transformed matrix
          - 1D array indexing is used to avoid confusion between row-major C & column-major Fortran (BLAS)
          - whenever possible, all matrix and/or vector operations are written as BLAS operations,
            even if there is no expectation of a performance benefit (e.g. BLAS level-1 operations)
          - with the standard 4-byte integer using in the BLAS and LAPACK interface,
            the largest square matrix that can be indexed properly is 46340-by-46340
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* generous machine precision used in accuracy tests */
#define EPS 1e-14

/* pairwise minimization & maximization macro */
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#define MAX(a, b) ((a) > (b)) ? (a) : (b)

/* column-major matrix indexing macro */
#define INDEX(row, col, stride) ((row) + (col)*(stride))

/* array offsetting macro */
#define OFFSET(array, offset) (&((array)[(offset)]))

/* TO DO:
   - implement & debug Cholesky QR
   - implement & debug modified Cholesky QR
*/

/* BLAS level-1 vector copy */
void dcopy(int*, double*, int*, double*, int*);

/* BLAS level-1 scalar-vector multiplication */
void dscal(int*, double*, double*, int*);

/* BLAS level-1 vector-vector inner product */
double ddot(int*, double*, int*, double*, int*);

/* BLAS level-2 vector-vector outer product */
void dger(int*, int*, double*, double*, int*, double*, int*, double*, int*);

/* BLAS level-2 matrix-vector multiplication */
double dgemv(char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

/* BLAS level-3 half-symmetric matrix-matrix multiplication */
void dsymm(char*, char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

/* BLAS level-3 triangular matrix inversion */
void dtrsm(char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int*);

/* BLAS level-3 symmetry-preserving matrix self-multiplication */
void dsyrk(char*, char*, int*, int*, double*, double*, int*, double*, double*, int*);

/* BLAS level-3 symmetry-preserving matrix-matrix multiplication */
void dsyr2k(char*, char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

/* BLAS level-3 matrix-matrix multiplication */
void dgemm(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

/* LAPACK symmetric eigensolver */
void dsyev(char*, char*, int*, double*, int*, double*, double*, int*, int*);

/* 1-sided (left side) application of a block Householder transformation: (I - W*Y^T)*M */
/* NOTE: this is similar to Algorithm 5 in the ELPA thesis */
/* M is a nrow-by-stride matrix */
/* A is a stride-by-ncol matrix workspace */
void householder_1sided(int nrow, int ncol, int stride, double *M, double *W, double *Y, double *A)
{
  char trans = 'T', notrans = 'N';
  double zero = 0.0, one = 1.0, minus_one = -1.0;

  /* A = M^T*Y */
  dgemm(&trans, &notrans, &stride, &ncol, &nrow, &one, M, &stride, Y, &stride, &zero, A, &stride);
  /* M = M - W*A^T */
  dgemm(&notrans, &trans, &nrow, &stride, &ncol, &minus_one, W, &stride, A, &stride, &one, M, &stride);
}

/* 2-sided application of a block Householder transformation: (I - W*Y^T)*M*(I - Y*W^T) */
/* NOTE: this is similar to Algorithm 7 in the ELPA thesis */
/* M is a symmetric nrow-by-nrow matrix, only the lower triangle is read & transformed */
/* A is a ncol-by-ncol matrix workspace & Z is a nrow-by-ncol matrix workspace */
void householder_2sided(int nrow, int ncol, int stride, double *M, double *W, double *Y, double *A, double *Z)
{
  char left = 'L', lo = 'L', trans = 'T', notrans = 'N';
  double zero = 0.0, one = 1.0, minus_one = -1.0, minus_half = -0.5;

  /* Z = M*Y */
  dsymm(&left, &lo, &nrow, &ncol, &one, M, &stride, Y, &stride, &zero, Z, &stride);
  /* A = Y^T*Z */
  dgemm(&trans, &notrans, &ncol, &ncol, &nrow, &one, Y, &stride, Z, &stride, &zero, A, &stride);
  /* Z = Z - 0.5*W*A */
  dgemm(&notrans, &notrans, &nrow, &ncol, &ncol, &minus_half, W, &stride, A, &stride, &one, Z, &stride);
  /* M = M - W*Z^T - Z*W^T */
  dsyr2k(&lo, &notrans, &nrow, &ncol, &minus_one, W, &stride, Z, &stride, &one, M, &stride);
}

/* Householder QR decomposition of a matrix stored as a block Householder transformation, Q = I - W*Y^T */
/* NOTE: this is similar to Algorithm 20 & unlabelled pseudocode on page 19 of the ELPA thesis */
/* M is overwritten by R upon return */
/* T is part of the compact representation: W = Y*T */
void qr_householder(int nrow, int ncol, int stride, double *M, double *W, double *Y, double *T)
{
  int i, j, rank = MIN(nrow, ncol);
  char trans = 'T', notrans = 'N', up = 'U', left = 'L', nounit = 'N';
  double zero = 0.0, half = 0.5, one = 1.0, minus_two = -2.0;

  for(i=0 ; i<rank ; i++)
  {
    int nrow2 = nrow-i, ncol2 = ncol-i, unit = 1;
    double wt, *M2 = OFFSET(M, INDEX(i,i,stride)), *Y2 = OFFSET(Y, INDEX(i,i,stride));

    /* calculate matrix element of reduced column */
    wt = sqrt(ddot(&nrow2, M2, &unit, M2, &unit));
    if(M2[0] > 0.0) { wt = -wt; }
    /* construct Householder vector in leading column of Y2 */
    dcopy(&nrow2, M2, &unit, Y2, &unit);
    Y2[0] -= wt;
    wt = 1.0/sqrt(2.0*wt*(wt - M2[0]));
    dscal(&nrow2, &wt, Y2, &unit);
    /* T = M2^T*Y2 (leading columns of T & Y2 only) */
    dgemv(&trans, &nrow2, &ncol2, &one, M2, &stride, Y2, &unit, &zero, T, &unit);
    /* M2 = M2 - 2.0*Y2*T^T (leading columns of T & Y2 only) */
    dger(&nrow2, &ncol2, &minus_two, Y2, &unit, T, &unit, M2, &stride);
    /* set unassigned matrix elements of Y to zero */
    for(j=0 ; j<i ; j++) { Y[INDEX(j,i,stride)] = 0.0; }
    /* clean the matrix elements formally removed from M */
    nrow2--;
    dscal(&nrow2, &zero, OFFSET(M2,1), &unit);
  }

  /* W = Y^T*Y (upper triangle only) */
  dsyrk(&up, &trans, &rank, &nrow, &one, Y, &stride, &zero, W, &stride);
  /* W *= 0.5 (diagonals only) */
  i = stride+1;
  dscal(&rank, &half, W, &i);
  /* T = W^{-1} */
  for(i=0 ; i<ncol ; i++) for(j=0 ; j<ncol ; j++) { T[INDEX(j,i,stride)] = 0.0; }
  for(i=0 ; i<ncol ; i++) { T[INDEX(i,i,stride)] = 1.0; }
  dtrsm(&left, &up, &notrans, &nounit, &rank, &rank, &one, W, &stride, T, &stride);
  /* W = Y*T */
  dgemm(&notrans, &notrans, &nrow, &rank, &rank, &one, Y, &stride, T, &stride, &zero, W, &stride);
}

/* Cholesky QR decomposition of a matrix stored as a block Householder transformation, Q = I - W*Y^T */
/* NOTE: this is equivalent to Algorithm 26 of the ELPA thesis, but calls LAPACK for Cholesky decomposition */
/* M is overwritten by R upon return */
/* T is part of the compact representation: W = Y*T */
void qr_cholesky(int nrow, int ncol, int stride, double *M, double *W, double *Y, double *T)
{
  /* T = M^T*M */
  /* Cholesky decomposition: T = R^T*R */
  /* ??? */
  /* profit */
}

/* print a matrix for debugging purposes */
void print_mat(int nrow, int ncol, int stride, double *mat)
{
  int i, j;
  for(i=0 ; i<ncol ; i++) for(j=0 ; j<nrow ; j++)
  { printf("%d %d %e\n",j,i,mat[INDEX(j,i,stride)]); }
}

/* full-to-band transformation of a test matrix & band-to-full transformation of its eigenvectors */
int main(int argc, char** argv)
{
  char jobv = 'V', lo = 'L';
  int i, j, mode, nblock, ndim, lwork = -1, info, unit = 1;
  double *mat1, *mat2, *hmat, *eval1, *eval2, *work, norm, kappa;

  /* read command-line inputs */
  if(argc < 4) { printf("USAGE: <executable> <mode> <block size> <matrix size>\n"); exit(1); }
  sscanf(argv[1],"%d",&mode); sscanf(argv[2],"%d",&nblock); sscanf(argv[3],"%d",&ndim);
  if(mode < 0 || mode > 1) { printf("ERROR: unknown mode (%d)\n", mode); exit(1); }

  /* workspace query */
  dsyev(&jobv, &lo, &ndim, mat1, &ndim, eval1, &kappa, &lwork, &info);
  lwork = MAX((int)kappa, nblock*ndim);

  /* allocate memory */
  mat1 = malloc(sizeof(double)*ndim*ndim);
  mat2 = malloc(sizeof(double)*ndim*ndim);
  hmat = malloc(sizeof(double)*ndim*ndim);
  eval1 = malloc(sizeof(double)*ndim);
  eval2 = malloc(sizeof(double)*ndim);
  work = malloc(sizeof(double)*lwork);

  /* construct lower triangle of an arbitrary test matrix */
  for(i=0 ; i<ndim ; i++) for(j=i ; j<ndim ; j++)
  { mat1[INDEX(j,i,ndim)] = mat2[INDEX(j,i,ndim)] = cos(i + sqrt(2.0)*j + sqrt(3.0)*ndim); }

  /* full-to-banded matrix transformation */
  for(i=nblock ; i<ndim ; i+=nblock)
  {
    /* submatrix dimensions & pointers to submatrices */
    int nrow = ndim-i;
    double *M = OFFSET(mat2, INDEX(i,i-nblock,ndim)),
           *M2 = OFFSET(mat2, INDEX(i,i,ndim)),
           *W = OFFSET(hmat, INDEX(0,ndim-i,ndim)),
           *Y = OFFSET(hmat, INDEX(i,i-nblock,ndim)), 
           *A = work,
           *Z = OFFSET(work, nblock);

    /* band reduction of column block with block Householder transformation */
    switch(mode)
    {
      case 0: { qr_householder(nrow, nblock, ndim, M, W, Y, A); break; }
      case 1: { qr_cholesky(nrow, nblock, ndim, M, W, Y, A); break; }
      default: { printf("ERROR: invalid mode (%d)\n", mode); exit(1); }
    }

    /* apply block Householder transformation to the rest of the matrix (2-sided) */
    householder_2sided(nrow, MIN(nrow,nblock), ndim, M2, Y, W, A, Z);
  }

  /* diagonalize the original & transformed matrices */
  dsyev(&jobv, &lo, &ndim, mat1, &ndim, eval1, work, &lwork, &info);
  dsyev(&jobv, &lo, &ndim, mat2, &ndim, eval2, work, &lwork, &info);

  /* back-transform the eigenvectors of the transformed matrix */
  for(i=ndim - ndim%nblock ; i>=nblock ; i-=nblock)
  {
    /* submatrix dimensions & pointers to submatrices */
    int nrow = ndim-i;
    double *M = OFFSET(mat2, i),
           *W = OFFSET(hmat, INDEX(0,ndim-i,ndim)),
           *Y = OFFSET(hmat, INDEX(i,i-nblock,ndim));

    /* apply block Householder transformation to the eigenvector matrix (1-sided) */
    householder_1sided(nrow, MIN(nrow,nblock), ndim, M, W, Y, work);
  }

  /* compare eigenvalues & eigenvectors to heuristic error bounds */
  norm = MAX(-eval1[0], eval1[ndim-1]);
  for(i=0 ; i<ndim ; i++)
  {
    double error, kappa, *vec1 = OFFSET(mat1, INDEX(0,i,ndim)), *vec2 = OFFSET(mat2, INDEX(0,i,ndim));

    error = fabs(eval1[i] - eval2[i]);
    if(error > norm*EPS) { printf("WARNING: large eigenvalue error (%d,%e > %e)\n", i, error, norm*EPS); }

    error = 1.0 - fabs(ddot(&ndim,vec1,&unit,vec2,&unit)) /
                  sqrt(ddot(&ndim,vec1,&unit,vec1,&unit) * ddot(&ndim,vec2,&unit,vec2,&unit));
    kappa = 1.0;
    if(i>0) { kappa = MAX(kappa, norm/(eval1[i] - eval1[i-1])); }
    if(i<ndim-1) { kappa = MAX(kappa, norm/(eval1[i+1] - eval1[i])); }
    if(error > kappa*EPS) { printf("WARNING: large eigenvector error (%d,%e > %e)\n", i, error, kappa*EPS); }
  }

  /* deallocate memory */
  free(work);
  free(eval2);
  free(eval1);
  free(hmat);
  free(mat2);
  free(mat1);
  return 1;
}
