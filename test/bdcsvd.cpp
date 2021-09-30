// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
// Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
// Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
// Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

// discard stack allocation as that too bypasses malloc
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#define EIGEN_RUNTIME_NO_MALLOC

#include "main.h"
#include <Eigen/SVD>
#include <iostream>
#include <Eigen/LU>


#define SVD_DEFAULT(M) BDCSVD<M>
#define SVD_FOR_MIN_NORM(M) BDCSVD<M>
#define SVD_STATIC_OPTIONS(M, O) BDCSVD<M, O>
#include "svd_common.h"



template<typename MatrixType>
void bdcsvd_method()
{
  enum { Size = MatrixType::RowsAtCompileTime };
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Matrix<RealScalar, Size, 1> RealVecType;
  MatrixType m = MatrixType::Identity();
  VERIFY_IS_APPROX(m.bdcSvd().singularValues(), RealVecType::Ones());
  VERIFY_RAISES_ASSERT(m.bdcSvd().matrixU());
  VERIFY_RAISES_ASSERT(m.bdcSvd().matrixV());
  VERIFY_IS_APPROX(m.template bdcSvd<ComputeFullU|ComputeFullV>().solve(m), m);
  VERIFY_IS_APPROX(m.template bdcSvd<ComputeFullU|ComputeFullV>().transpose().solve(m), m);
  VERIFY_IS_APPROX(m.template bdcSvd<ComputeFullU|ComputeFullV>().adjoint().solve(m), m);
}

// Compare the Singular values returned with Jacobi and Bdc.
template<typename MatrixType, int Options>
void compare_bdc_jacobi(const MatrixType& a = MatrixType(), unsigned int computationOptions = 0, int algoswap = 16, bool random = true)
// compare the Singular values returned with Jacobi and Bdc
{
  MatrixType m = random ? MatrixType::Random(a.rows(), a.cols()) : a;

  BDCSVD<MatrixType, Options> bdc_svd(m.rows(), m.cols());
  bdc_svd.setSwitchSize(algoswap);
  bdc_svd.compute(m);

  JacobiSVD<MatrixType, Options> jacobi_svd(m);
  VERIFY_IS_APPROX(bdc_svd.singularValues(), jacobi_svd.singularValues());

  if(computationOptions & ComputeFullU) VERIFY_IS_APPROX(bdc_svd.matrixU(), jacobi_svd.matrixU());
  if(computationOptions & ComputeThinU) VERIFY_IS_APPROX(bdc_svd.matrixU(), jacobi_svd.matrixU());
  if(computationOptions & ComputeFullV) VERIFY_IS_APPROX(bdc_svd.matrixV(), jacobi_svd.matrixV());
  if(computationOptions & ComputeThinV) VERIFY_IS_APPROX(bdc_svd.matrixV(), jacobi_svd.matrixV());
}

// Verifies total deflation is **not** triggered.
void compare_bdc_jacobi_instance(bool structure_as_m, int algoswap = 16)
{
  MatrixXd m(4, 3);
  if (structure_as_m) {
    // The first 3 rows are the reduced form of Matrix 1 as shown below, and it
    // has nonzero elements in the first column and diagonals only.
    m << 1.056293, 0, 0,
         -0.336468, 0.907359, 0,
         -1.566245, 0, 0.149150,
         -0.1, 0, 0;
  } else {
    // Matrix 1.
    m << 0.882336, 18.3914, -26.7921,
         -5.58135, 17.1931, -24.0892,
         -20.794, 8.68496, -4.83103,
         -8.4981, -10.5451, 23.9072;
  }
  compare_bdc_jacobi(m, 0, algoswap, false);
}

template<typename MatrixType>
void bdcsvd_static_all_options(const MatrixType& input = MatrixType())
{
  MatrixType m = input;
  svd_fill_random(m);
  svd_static_option_checks<MatrixType, 0>(m);
}

EIGEN_DECLARE_TEST(bdcsvd)
{
  CALL_SUBTEST_3(( svd_static_verify_assert<Matrix3f>() ));
  CALL_SUBTEST_4(( svd_static_verify_assert<Matrix4d>() ));
  CALL_SUBTEST_7(( svd_static_verify_assert<Matrix<float, 30, 21> >() ));
  CALL_SUBTEST_7(( svd_static_verify_assert<Matrix<float, 21, 30> >() ));
  CALL_SUBTEST_9(( svd_static_verify_assert<Matrix<std::complex<double>, 20, 27> >() ));

  CALL_SUBTEST_101(( svd_all_trivial_2x2(bdcsvd_static_all_options<Matrix2cd>) ));
  CALL_SUBTEST_102(( svd_all_trivial_2x2(bdcsvd_static_all_options<Matrix2d>) ));

  for(int i = 0; i < g_repeat; i++) {
    int r = internal::random<int>(1, EIGEN_TEST_MAX_SIZE/2),
        c = internal::random<int>(1, EIGEN_TEST_MAX_SIZE/2);

    TEST_SET_BUT_UNUSED_VARIABLE(r)
    TEST_SET_BUT_UNUSED_VARIABLE(c)

    CALL_SUBTEST_7((  compare_bdc_jacobi<MatrixXf, ComputeFullU | ComputeThinV>(MatrixXf(r,c)) ));
    CALL_SUBTEST_10(( compare_bdc_jacobi<MatrixXd, ComputeFullU | ComputeFullV>(MatrixXd(r,c)) ));
    CALL_SUBTEST_8((  compare_bdc_jacobi<MatrixXcd, ComputeThinU | ComputeThinV>(MatrixXcd(r,c)) ));
    // Test on inf/nan matrix
    CALL_SUBTEST_7(  (svd_inf_nan<MatrixXf>()) );
    CALL_SUBTEST_10( (svd_inf_nan<MatrixXd>()) );

    // Verify some computations using all combinations of the Options template parameter.
    CALL_SUBTEST_3(( bdcsvd_static_all_options<Matrix3f>() ));
    CALL_SUBTEST_3(( bdcsvd_static_all_options<Matrix<float, 2, 3> >() ));
    // CALL_SUBTEST_4(( bdcsvd_static_all_options<Matrix<double, 20, 17> >() ));
    // CALL_SUBTEST_4(( bdcsvd_static_all_options<Matrix<double, 17, 20> >() ));
    CALL_SUBTEST_5(( bdcsvd_static_all_options<Matrix<double, Dynamic, 30> >(Matrix<double, Dynamic, 30>(r, 30)) ));
    CALL_SUBTEST_5(( bdcsvd_static_all_options<Matrix<double, 20, Dynamic> >(Matrix<double, 20, Dynamic>(20, c)) ));
    CALL_SUBTEST_7(( bdcsvd_static_all_options<MatrixXf>(MatrixXf(r, c)) ));
    CALL_SUBTEST_8(( bdcsvd_static_all_options<MatrixXcd>(MatrixXcd(r, c)) ));
    CALL_SUBTEST_10(( bdcsvd_static_all_options<MatrixXd>(MatrixXd(r, c)) ));
    CALL_SUBTEST_14(( bdcsvd_static_all_options<Matrix<double, 20, 27, RowMajor>>() ));
    CALL_SUBTEST_14(( bdcsvd_static_all_options<Matrix<double, 27, 20, RowMajor>>() ));

    CALL_SUBTEST_15(( svd_check_max_size_matrix< Matrix<float, Dynamic, Dynamic, ColMajor, 20, 35>, ColPivHouseholderQRPreconditioner >(r, c) ));
    CALL_SUBTEST_15(( svd_check_max_size_matrix< Matrix<float, Dynamic, Dynamic, ColMajor, 35, 20>, HouseholderQRPreconditioner >(r, c) ));
    CALL_SUBTEST_15(( svd_check_max_size_matrix< Matrix<float, Dynamic, Dynamic, RowMajor, 20, 35>, ColPivHouseholderQRPreconditioner >(r, c) ));
    CALL_SUBTEST_15(( svd_check_max_size_matrix< Matrix<float, Dynamic, Dynamic, RowMajor, 35, 20>, HouseholderQRPreconditioner >(r, c) ));
  }

  // test matrixbase method
  CALL_SUBTEST_1(( bdcsvd_method<Matrix2cd>() ));
  CALL_SUBTEST_3(( bdcsvd_method<Matrix3f>() ));

  // Test problem size constructors
  CALL_SUBTEST_7( BDCSVD<MatrixXf>(10,10) );

  // Check that preallocation avoids subsequent mallocs
  // Disabled because not supported by BDCSVD
  // CALL_SUBTEST_9( svd_preallocate<void>() );

  CALL_SUBTEST_2( svd_underoverflow<void>() );

  // Without total deflation issues.
  CALL_SUBTEST_11((  compare_bdc_jacobi_instance(true) ));
  CALL_SUBTEST_12((  compare_bdc_jacobi_instance(false) ));

  // With total deflation issues before, when it shouldn't be triggered.
  CALL_SUBTEST_13((  compare_bdc_jacobi_instance(true, 3) ));
  CALL_SUBTEST_14((  compare_bdc_jacobi_instance(false, 3) ));
}
