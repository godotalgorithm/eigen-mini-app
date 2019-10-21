/* Eigenrockets mini-app for QR decomposition: serial implementation

   USAGE: <executable> <mode> <nrow> <ncol>
   <mode> : 0 for Householder QR algorithm
            1 for Cholesky QR algorithm (unstable)
            2 for modified Cholesky QR algorithm (not implemented yet)
            3 for LAPACK's standard QR algorithm
   <nrow> : number of rows in the test matrix
   <ncol> : number of columns in the test matrix

   NOTES: - for simplicity & rapidity of development, this is written in ANSI C (C89)
            and uses the standard Fortran interfaces to BLAS & LAPACK
          - this implementation is restricted to real doubles
          - block Householder transformations are stored as T & Y in Q = I - Y*T*Y^T
          - 1D array indexing is used to avoid confusion between row-major C & column-major Fortran (BLAS)
          - whenever possible, all matrix and/or vector operations are written as BLAS & LAPACK operations,
            even if there is little expectation of a performance benefit (e.g. BLAS level-1 operations)
          - the Cholesky QR algorithm is a "block" QR algorithm where R has a block size of ncol,
            so actually it is a square matrix and not an upper triangular matrix at all
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* column-major matrix indexing macro */
#define INDEX(row, col, stride) ((row) + (col)*(stride))

/* array offsetting macro */
#define OFFSET(array, offset) (&((array)[(offset)]))

/* BLAS level-1 vector swap */
void dswap_(int*, double*, int*, double*, int*);

/* BLAS level-1 scalar-vector multiplication */
void dscal_(int*, double*, double*, int*);

/* BLAS level-1 vector-vector inner product */
double ddot_(int*, double*, int*, double*, int*);

/* BLAS level-1 vector addition */
void daxpy_(int*, double*, double*, int*, double*, int*);

/* BLAS level-2 vector-vector outer product */
void dger_(int*, int*, double*, double*, int*, double*, int*, double*, int*);

/* BLAS level-2 matrix-vector multiplication */
double dgemv_(char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

/* BLAS level-3 triangular matrix multiplication */
void dtrmm_(char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int*);

/* BLAS level-3 triangular matrix inversion */
void dtrsm_(char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int*);

/* BLAS level-3 symmetry-preserving matrix self-multiplication */
void dsyrk_(char*, char*, int*, int*, double*, double*, int*, double*, double*, int*);

/* BLAS level-3 matrix-matrix multiplication */
void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

/* LAPACK matrix initialization */
void dlaset_(char*, int*, int*, double*, double*, double*, int*);

/* LAPACK matrix copy */
void dlacpy_(char*, int*, int*, double*, int*, double*, int*);

/* LAPACK Cholesky decomposition */
void dpotrf_(char*, int*, double*, int*, int*);

/* LAPACK QR decomposition */
void dgeqrf_(int*, int*, double*, int*, double*, double*, int*, int*);

/* LAPACK accumulation of Householder rotations */
void dorgqr_(int*, int*, int*, double*, int*, double*, double*, int*, int*);

/* print a matrix for debugging purposes */
void print_mat(int nrow, int ncol, int stride, double *mat)
{
  int i, j;
  for(i=0 ; i<ncol ; i++) for(j=0 ; j<nrow ; j++)
  { printf("%d %d %e\n",j,i,mat[INDEX(j,i,stride)]); }
}

/* in-place block Householder transformation of a matrix: (I - Y*op(T)*Y^T)*M */
/* T is an upper-triangular matrix, op(T) = T for type = 'N' & op(T) = T^T for type = 'T' */
/* work has a size of rank*ncol */
void transform(char type, int nrow, int ncol, int rank, double *T, double *Y, double *M, double *work)
{
  char trans = 'T', notrans = 'N', left = 'L', up = 'U', nounit = 'N';
  double zero = 0.0, one = 1.0, minus_one = -1.0;

  /* work = Y^T*M */
  dgemm_(&trans, &notrans, &rank, &ncol, &nrow, &one, Y, &nrow, M, &nrow, &zero, work, &rank);
  /* work = op(T)*work */
  dtrmm_(&left, &up, &type, &nounit, &rank, &ncol, &one, T, &rank, work, &rank);
  /* M = M - Y*work */
  dgemm_(&notrans, &notrans, &nrow, &ncol, &rank, &minus_one, Y, &nrow, work, &rank, &one, M, &nrow);
}

/* Householder QR decomposition of a matrix, M = Q*R */
/* nrow >= ncol is assumed */
/* Q is stored as a block Householder transformation, Q = I - Y*T*Y^T */
/* M is overwritten by Y upon return */
/* work has size ncol*ncol */
void qr_householder(int nrow, int ncol, double *M, double *T, double *R, double *work)
{
  int i, j, unit = 1;
  char all = 'A', trans = 'T', notrans = 'N', up = 'U', left = 'L', nounit = 'N';
  double zero = 0.0, half = 0.5, one = 1.0, minus_two = -2.0;

  /* clear R */
  dlaset_(&all, &ncol, &ncol, &zero, &zero, R, &ncol);

  for(i=0 ; i<ncol ; i++)
  {
    int nrow2 = nrow-i, ncol2 = ncol-i-1;
    double wt, *Mdiag = OFFSET(M, INDEX(i,i,nrow)), *Mnext = OFFSET(M, INDEX(i,i+1,nrow)),
           *Mcol = OFFSET(M, INDEX(0,i,nrow)), *Rcol = OFFSET(R, INDEX(0,i,ncol));

    /* calculate matrix element of reduced column */
    wt = sqrt(ddot_(&nrow2, Mdiag, &unit, Mdiag, &unit));
    if(Mdiag[0] > 0.0) { wt = -wt; }
    /* assign upper triangle of Y & R */
    dswap_(&i, Mcol, &unit, Rcol, &unit);
    Rcol[i] = wt;
    /* construct next Householder vector */
    wt = 1.0/sqrt(2.0*wt*(wt - Mdiag[0]));
    Mdiag[0] -= Rcol[i];
    dscal_(&nrow2, &wt, Mdiag, &unit);
    /* work = Mnext^T*Mdiag (leading column Mdiag only) */
    dgemv_(&trans, &nrow2, &ncol2, &one, Mnext, &nrow, Mdiag, &unit, &zero, work, &unit);
    /* Mnext = Mnext - 2.0*Mdiag*work^T (leading columns of Mdiag only) */
    dger_(&nrow2, &ncol2, &minus_two, Mdiag, &unit, work, &unit, Mnext, &nrow);
  }

  /* work = M^T*M (upper triangle only) */
  dsyrk_(&up, &trans, &ncol, &nrow, &one, M, &nrow, &zero, work, &ncol);
  /* work *= 0.5 (diagonals only) */
  i = ncol+1;
  dscal_(&ncol, &half, work, &i);
  /* T = work^{-1} */
  dlaset_(&all, &ncol, &ncol, &zero, &one, T, &ncol);
  dtrsm_(&left, &up, &notrans, &nounit, &ncol, &ncol, &one, work, &ncol, T, &ncol);
}

/* Cholesky QR decomposition of a matrix, M = Q*R */
/* nrow >= ncol is assumed (& some even stricter memory constraints because I'm lazy) */
/* Q is stored as a block Householder transformation, Q = I - Y*T*Y^T */
/* M is overwritten by Y upon return */
/* work has size ncol*nrow */
void qr_cholesky(int nrow, int ncol, double *M, double *T, double *R, double *work)
{
  int i, lwork = ncol*(nrow - ncol - 1), info, nmat = ncol*ncol, unit = 1;
  char all = 'A', up = 'U', trans = 'T', notrans = 'N', left = 'L', right = 'R', nounit = 'N';
  double zero = 0.0, one = 1.0, minus_one = -1.0, *wvec = OFFSET(work, ncol*ncol), *work2 = OFFSET(work, ncol*(ncol+1));
  if(lwork < ncol) { printf("ERROR: not enough QR workspace, make nrow bigger\n"); }

  /* T = M^T*M */
  dsyrk_(&up, &trans, &ncol, &nrow, &one, M, &nrow, &zero, T, &ncol);
  /* Cholesky decomposition: U -> T for T = U^T*U */
  dpotrf_(&up, &ncol, T, &ncol, &info);
  if(info != 0) { printf("ERROR: Cholesky decomposition failure (%d)\n", info); exit(1); }
  /* M = M*T^{-1} */
  dtrsm_(&right, &up, &notrans, &nounit, &nrow, &ncol, &one, T, &ncol, M, &nrow);
  /* QR decomposition: R -> work & Q -> R for M(top ncol-by-ncol block) = Q*R */
  dlacpy_(&all, &ncol, &ncol, M, &nrow, R, &ncol);
  dgeqrf_(&ncol, &ncol, R, &ncol, wvec, work2, &lwork, &info);
  dlacpy_(&up, &ncol, &ncol, R, &ncol, work, &ncol);
  dorgqr_(&ncol, &ncol, &ncol, R, &ncol, wvec, work2, &lwork, &info);
  for(i=0 ; i<ncol ; i++)
  {
    if(work[INDEX(i,i,ncol)] < 0.0)
    {
      dscal_(&ncol, &minus_one, OFFSET(work,i), &ncol);
      dscal_(&ncol, &minus_one, OFFSET(R,INDEX(0,i,ncol)), &unit);
    }
  }
  /* M = M + R */
  for(i=0 ; i<ncol ; i++)
  { daxpy_(&ncol, &one, OFFSET(R,INDEX(0,i,ncol)), &unit, OFFSET(M,INDEX(0,i,nrow)), &unit); }
  /* R = -R*T */
  dtrmm_(&right, &up, &notrans, &nounit, &ncol, &ncol, &minus_one, T, &ncol, R, &ncol);
  /* T = (I + work)^{-1} */
  dlaset_(&all, &ncol, &ncol, &zero, &one, T, &ncol);
  daxpy_(&nmat, &one, T, &unit, work, &unit);
  dtrsm_(&left, &up, &notrans, &nounit, &ncol, &ncol, &one, work, &ncol, T, &ncol);
}

/* LAPACK's standard QR decomposition of a matrix, M = Q*R */
/* nrow >= ncol is assumed */
/* Q is stored as a block Householder transformation, Q = I - Y*T*Y^T */
/* M is overwritten by Y upon return */
/* work has size nrow*ncol */
void qr_lapack(int nrow, int ncol, double *M, double *T, double *R, double *work)
{
  int i, j, unit = 1, lwork = nrow*ncol, info;
  char all = 'A', trans = 'T', notrans = 'N', up = 'U', left = 'L', nounit = 'N';
  double zero = 0.0, half = 0.5, one = 1.0, minus_two = -2.0;

  /* QR decomposition */
  dgeqrf_(&nrow, &ncol, M, &nrow, T, work, &lwork, &info);

  /* extract R */
  dlaset_(&all, &ncol, &ncol, &zero, &one, R, &ncol);
  for(i=1 ; i<=ncol ; i++)
  { dswap_(&i, OFFSET(M, INDEX(0,i-1,nrow)), &unit, OFFSET(R, INDEX(0,i-1,ncol)), &unit); }

  /* work = M^T*M (upper triangle only) */
  dsyrk_(&up, &trans, &ncol, &nrow, &one, M, &nrow, &zero, work, &ncol);
  /* work *= 0.5 (diagonals only) */
  i = ncol+1;
  dscal_(&ncol, &half, work, &i);
  /* T = work^{-1} */
  dlaset_(&all, &ncol, &ncol, &zero, &one, T, &ncol);
  dtrsm_(&left, &up, &notrans, &nounit, &ncol, &ncol, &one, work, &ncol, T, &ncol);
}

/* QR decomposition of a random-ish matrix & accuracy tests */
int main(int argc, char** argv)
{
  char all = 'A', type;
  int i, j, mode, nrow, ncol, lwork = -1, info, unit = 1;
  double *Y, *R, *T, *work, *work2, zero = 0.0, error;     
  clock_t start, end;

  /* read command-line inputs */
  if(argc < 4) { printf("USAGE: <executable> <mode> <nrow> <ncol>\n"); exit(1); }
  sscanf(argv[1],"%d",&mode); sscanf(argv[2],"%d",&nrow); sscanf(argv[3],"%d",&ncol);
  if(nrow < ncol) { printf("ERROR: nrow < ncol\n"); exit(1); }

  /* allocate memory */
  Y = malloc(sizeof(double)*nrow*ncol);
  R = malloc(sizeof(double)*ncol*ncol);
  T = malloc(sizeof(double)*ncol*ncol);
  work = malloc(sizeof(double)*nrow*ncol);
  work2 = malloc(sizeof(double)*ncol*ncol);

  /* construct test matrix */
  srand(1);
  for(i=0 ; i<ncol ; i++) for(j=0 ; j<nrow ; j++)
  { Y[INDEX(j,i,nrow)] = 2.0*(double)rand()/(double)RAND_MAX - 1.0; }

  /* QR decomposition */
  start = clock();
  switch(mode)
  {
    case 0: { qr_householder(nrow, ncol, Y, T, R, work); type = 'N'; break; }
    case 1: { qr_cholesky(nrow, ncol, Y, T, R, work); type = 'T'; break; }
    case 3: { qr_lapack(nrow, ncol, Y, T, R, work); type = 'N'; break; }
    default: { printf("ERROR: unknown mode (%d not in {0,1,3})\n", mode); exit(1); }
  }
  end = clock();
  printf("QR time = %e s\n", ((double) (end - start)) / CLOCKS_PER_SEC);
//printf("R:\n");
//print_mat(ncol,ncol,ncol,R);

  /* reconstruction test */
  dlaset_(&all, &nrow, &ncol, &zero, &zero, work, &nrow);
  dlacpy_(&all, &ncol, &ncol, R, &ncol, work, &nrow);
  start = clock();
  transform(type, nrow, ncol, ncol, T, Y, work, work2);
  end = clock();
  printf("transform time = %e s\n", ((double) (end - start)) / CLOCKS_PER_SEC);

  /* measure reconstruction error */
  error = 0.0;
  srand(1);
  for(i=0 ; i<ncol ; i++) for(j=0 ; j<nrow ; j++)
  {
    double new_error = fabs(work[INDEX(j,i,nrow)] - (2.0*(double)rand()/(double)RAND_MAX - 1.0));
    if(new_error > error) { error = new_error; }
  }
  printf("reconstruction error = %e\n", error);

  /* unitary test */
  srand(1);
  for(i=0 ; i<ncol ; i++) for(j=0 ; j<nrow ; j++)
  { work[INDEX(j,i,nrow)] = 2.0*(double)rand()/(double)RAND_MAX - 1.0; }
  transform('N', nrow, ncol, ncol, T, Y, work, work2);
  transform('T', nrow, ncol, ncol, T, Y, work, work2);

  /* measure unitary error */
  error = 0.0;
  srand(1);
  for(i=0 ; i<ncol ; i++) for(j=0 ; j<nrow ; j++)
  {
    double new_error = fabs(work[INDEX(j,i,nrow)] - (2.0*(double)rand()/(double)RAND_MAX - 1.0));
    if(new_error > error) { error = new_error; }
  }
  printf("unitary error = %e\n", error);

  /* deallocate memory */
  free(work2);
  free(work);
  free(T);
  free(R);
  free(Y);
  return 1;
}
