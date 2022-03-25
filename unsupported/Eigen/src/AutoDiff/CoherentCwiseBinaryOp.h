// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 The Eigen Team.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COHERENT_CWISE_BINARY_OP_H
#define EIGEN_COHERENT_CWISE_BINARY_OP_H

#include "./InternalHeaderCheck.h"

namespace Eigen {

template<typename BinaryOp,  typename Lhs, typename Rhs>  class CoherentCwiseBinaryOp;

namespace internal {

template<typename BinaryOp, typename Lhs, typename Rhs>
struct traits<CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs> > 
  : public traits<CwiseBinaryOp<BinaryOp, Lhs, Rhs>> {};

} // end namespace internal

template<typename BinaryOp, typename Lhs, typename Rhs, typename StorageKind>
class CoherentCwiseBinaryOpImpl;

/** \class CoherentCwiseBinaryOp
  *
  * \brief Generic expression where a coefficient-wise binary operator is applied to two expressions
  * If one expression has zero size, it is artificially padded with zeros to match the size of the other expression.
  *
  * \tparam BinaryOp template functor implementing the operator
  * \tparam LhsType the type of the left-hand side
  * \tparam RhsType the type of the right-hand side
  *
  * This class represents an expression  where a coefficient-wise binary operator is applied to two expressions.
  * It is the return type of binary operators, by which we mean only those binary operators where
  * both the left-hand side and the right-hand side are Eigen expressions.
  *
  * Most of the time, this is the only way that it is used, so you typically don't have to name
  * CoherentCwiseBinaryOp types explicitly.
  */
template<typename BinaryOp, typename LhsType, typename RhsType>
class CoherentCwiseBinaryOp :
  public CoherentCwiseBinaryOpImpl<
          BinaryOp, LhsType, RhsType,
          typename internal::cwise_promote_storage_type<typename internal::traits<LhsType>::StorageKind,
                                                        typename internal::traits<RhsType>::StorageKind,
                                                        BinaryOp>::ret>,
  internal::no_assignment_operator
{
  public:

    typedef typename internal::remove_all<BinaryOp>::type Functor;
    typedef typename internal::remove_all<LhsType>::type Lhs;
    typedef typename internal::remove_all<RhsType>::type Rhs;

    typedef typename CoherentCwiseBinaryOpImpl<
        BinaryOp, LhsType, RhsType,
        typename internal::cwise_promote_storage_type<typename internal::traits<LhsType>::StorageKind,
                                                      typename internal::traits<Rhs>::StorageKind,
                                                      BinaryOp>::ret>::Base Base;
    EIGEN_GENERIC_PUBLIC_INTERFACE(CoherentCwiseBinaryOp)

    EIGEN_CHECK_BINARY_COMPATIBILIY(BinaryOp,typename Lhs::Scalar,typename Rhs::Scalar)
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Lhs, Rhs)

    typedef typename internal::ref_selector<LhsType>::type LhsNested;
    typedef typename internal::ref_selector<RhsType>::type RhsNested;
    typedef typename std::remove_reference<LhsNested>::type LhsNested_;
    typedef typename std::remove_reference<RhsNested>::type RhsNested_;
    typedef typename internal::remove_all<LhsNested>::type PlainLhs;
    typedef typename internal::remove_all<RhsNested>::type PlainRhs;
    
    enum {
      LhsRowsAtCompileTime = internal::traits<PlainLhs>::RowsAtCompileTime,
      LhsColsAtCompileTime = internal::traits<PlainLhs>::ColsAtCompileTime,
      LhsSizeAtCompileTime = (LhsRowsAtCompileTime==Dynamic || LhsColsAtCompileTime==Dynamic) ? Dynamic : LhsRowsAtCompileTime * LhsColsAtCompileTime,
      RhsRowsAtCompileTime = internal::traits<PlainRhs>::RowsAtCompileTime,
      RhsColsAtCompileTime = internal::traits<PlainRhs>::ColsAtCompileTime,
      RhsSizeAtCompileTime = (RhsRowsAtCompileTime==Dynamic || RhsColsAtCompileTime==Dynamic) ? Dynamic : RhsRowsAtCompileTime * RhsColsAtCompileTime
    };
    
#if EIGEN_COMP_MSVC
    //Required for Visual Studio or the Copy constructor will probably not get inlined!
    EIGEN_STRONG_INLINE
    CoherentCwiseBinaryOp(const CoherentCwiseBinaryOp<BinaryOp,LhsType,RhsType>&) = default;
#endif

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    CoherentCwiseBinaryOp(const Lhs& aLhs, const Rhs& aRhs, const BinaryOp& func = BinaryOp())
      : m_lhs(aLhs), m_rhs(aRhs), m_functor(func)
    {
      // Purposely do not assert, since we are specifically dealing with inputs of different sizes.
      // eigen_assert(aLhs.rows() == aRhs.rows() && aLhs.cols() == aRhs.cols());
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR
    Index rows() const EIGEN_NOEXCEPT {
      // return the fixed size type if available to enable compile time optimizations
      return LhsRowsAtCompileTime != Dynamic && RhsRowsAtCompileTime != Dynamic ? std::max<Index>(LhsRowsAtCompileTime, RhsRowsAtCompileTime)
                                                                                : std::max<Index>(m_lhs.rows(), m_rhs.rows());
    }
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR
    Index cols() const EIGEN_NOEXCEPT {
      // return the fixed size type if available to enable compile time optimizations
      return LhsColsAtCompileTime != Dynamic && RhsColsAtCompileTime != Dynamic ? std::max<Index>(LhsColsAtCompileTime, RhsColsAtCompileTime)
                                                                                : std::max<Index>(m_lhs.cols(), m_rhs.cols());
    }

    /** \returns the left hand side nested expression */
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const LhsNested_& lhs() const { return m_lhs; }
    /** \returns the right hand side nested expression */
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const RhsNested_& rhs() const { return m_rhs; }
    /** \returns the functor representing the binary operation */
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const BinaryOp& functor() const { return m_functor; }

  protected:
    LhsNested m_lhs;
    RhsNested m_rhs;
    const BinaryOp m_functor;
};

// Generic API dispatcher
template<typename BinaryOp, typename Lhs, typename Rhs, typename StorageKind>
class CoherentCwiseBinaryOpImpl
  : public internal::generic_xpr_base<CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs> >::type
{
public:
  typedef typename internal::generic_xpr_base<CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs> >::type Base;
};

// -------------------- Evaluator --------------------

namespace internal {

template<typename BinaryOp, typename Lhs, typename Rhs>
struct evaluator<CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs> >
  : public binary_evaluator<CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs> >
{
  typedef CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs> XprType;
  typedef binary_evaluator<CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs> > Base;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  explicit evaluator(const XprType& xpr) : Base(xpr) {}
};

template<typename BinaryOp, typename Lhs, typename Rhs>
struct binary_evaluator<CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs>, IndexBased, IndexBased>
  : evaluator_base<CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs> >
{
  typedef CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs> XprType;
  using LhsScalar = typename Lhs::Scalar;
  using RhsScalar = typename Rhs::Scalar;

  enum {
    CoeffReadCost = int(evaluator<Lhs>::CoeffReadCost) + int(evaluator<Rhs>::CoeffReadCost) + int(functor_traits<BinaryOp>::Cost),

    LhsFlags = evaluator<Lhs>::Flags,
    RhsFlags = evaluator<Rhs>::Flags,
    SameType = is_same<LhsScalar, RhsScalar>::value,
    StorageOrdersAgree = (int(LhsFlags)&RowMajorBit)==(int(RhsFlags)&RowMajorBit),
    Flags0 = (int(LhsFlags) | int(RhsFlags)) & (
        HereditaryBits
      | (int(LhsFlags) & int(RhsFlags) &
           ( (StorageOrdersAgree ? LinearAccessBit : 0)
           | (functor_traits<BinaryOp>::PacketAccess && StorageOrdersAgree && SameType ? PacketAccessBit : 0)
           )
        )
     ),
    Flags = (Flags0 & ~RowMajorBit) | (LhsFlags & RowMajorBit),
    Alignment = plain_enum_min(evaluator<Lhs>::Alignment, evaluator<Rhs>::Alignment)
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  explicit binary_evaluator(const XprType& xpr) : m_d(xpr)
  {
    EIGEN_INTERNAL_CHECK_COST_VALUE(functor_traits<BinaryOp>::Cost);
    EIGEN_INTERNAL_CHECK_COST_VALUE(CoeffReadCost);
  }

  typedef typename XprType::CoeffReturnType CoeffReturnType;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  CoeffReturnType coeff(Index row, Index col) const
  {
    // Check if sizes are different.  This allows some compile-time
    // optimizations in the case that they are both fixed and equal-sized.
    if (m_d.lhsRows.value() != m_d.rhsRows.value() || m_d.lhsCols.value() != m_d.rhsCols.value()) {
      LhsScalar left = LhsScalar(0);
      if (row < m_d.lhsRows.value() && col < m_d.lhsCols.value()) {
        left = m_d.lhsImpl.coeff(row, col);
      }
      RhsScalar right = RhsScalar(0);
      if (row < m_d.rhsRows.value() && col < m_d.rhsCols.value()) {
        right = m_d.rhsImpl.coeff(row, col);
      }
      return m_d.func()(left, right);
    }
     
    // LHS and RHS are the same size, call naively.
    return m_d.func()(m_d.lhsImpl.coeff(row, col), m_d.rhsImpl.coeff(row, col));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  CoeffReturnType coeff(Index index) const
  {
    // Check if sizes are different.  This allows some compile-time
    // optimizations in the case that they are both fixed and equal-sized.
    if (m_d.lhsSize.value() != m_d.rhsSize.value()) {
      LhsScalar left = LhsScalar(0);
      if (index < m_d.lhsSize.value()) {
        left = m_d.lhsImpl.coeff(index);
      }
      RhsScalar right = RhsScalar(0);
      if (index < m_d.rhsSize.value()) {
        right = m_d.rhsImpl.coeff(index);
      }
      return m_d.func()(left, right);
    }
     
    // LHS and RHS are the same size, call naively.
    return m_d.func()(m_d.lhsImpl.coeff(index), m_d.rhsImpl.coeff(index));
  }

  template<int LoadMode, typename PacketType>
  EIGEN_STRONG_INLINE
  PacketType packet(Index row, Index col) const
  {
    // Check if sizes are different.  This allows some compile-time
    // optimizations in the case that they are both fixed and equal-sized.
    if (m_d.lhsRows.value() != m_d.rhsRows.value() || m_d.lhsCols.value() != m_d.rhsCols.value()) {
      
      // TODO(cantonios) Double-check we can actually load a packet from LHS/RHS,
      // otherwise we need a masked load.
      if (row < m_d.lhsRows.value() && col < m_d.lhsCols.value()) {
        const PacketType left = m_d.lhsImpl.template packet<LoadMode,PacketType>(row, col);
        if (row < m_d.rhsRows.value() && col < m_d.rhsCols.value()) {
          const PacketType right = m_d.rhsImpl.template packet<LoadMode,PacketType>(row, col);
          return m_d.func().packetOp(left, right);
        } else {
          const PacketType right = pzero(left);
          return m_d.func().packetOp(left, right);
        }
      } else {
        if (row < m_d.rhsRows.value() && col < m_d.rhsCols.value()) {
          const PacketType right = m_d.rhsImpl.template packet<LoadMode,PacketType>(row, col);
          const PacketType left = pzero(right);
          return m_d.func().packetOp(left, right);
        } else {
          const PacketType left = pset1<PacketType>(LhsScalar(0));
          const PacketType right = pzero(left);
          return m_d.func().packetOp(left, right);
        }
      }
    }
    return m_d.func().packetOp(m_d.lhsImpl.template packet<LoadMode,PacketType>(row, col),
                               m_d.rhsImpl.template packet<LoadMode,PacketType>(row, col));
  }

  template<int LoadMode, typename PacketType>
  EIGEN_STRONG_INLINE
  PacketType packet(Index index) const
  {
    // Check if sizes are different.  This allows some compile-time
    // optimizations in the case that they are both fixed and equal-sized.
    if (m_d.lhsSize.value() != m_d.rhsSize.value()) {
      
      // TODO(cantonios) Double-check we can actually load a packet from LHS/RHS,
      // otherwise we need a masked load.
      if (index < m_d.lhsSize.value()) {
        const PacketType left = m_d.lhsImpl.template packet<LoadMode,PacketType>(index);
        if (index < m_d.rhsSize.value()) {
          const PacketType right = m_d.rhsImpl.template packet<LoadMode,PacketType>(index);
          return m_d.func().packetOp(left, right);
        } else {
          const PacketType right = pzero(left);
          return m_d.func().packetOp(left, right);
        }
      } else {
        if (index < m_d.rhsSize.value()) {
          const PacketType right = m_d.rhsImpl.template packet<LoadMode,PacketType>(index);
          const PacketType left = pzero(right);
          return m_d.func().packetOp(left, right);
        } else {
          const PacketType left = pset1<PacketType>(LhsScalar(0));
          const PacketType right = pzero(left);
          return m_d.func().packetOp(left, right);
        }
      }
    }
    return m_d.func().packetOp(m_d.lhsImpl.template packet<LoadMode,PacketType>(index),
                               m_d.rhsImpl.template packet<LoadMode,PacketType>(index));
  }

protected:

  // this helper permits to completely eliminate the functor if it is empty
  struct Data
  {
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Data(const XprType& xpr) : op(xpr.functor()), lhsImpl(xpr.lhs()), rhsImpl(xpr.rhs()),
                               lhsRows(xpr.lhs().rows()), lhsCols(xpr.lhs().cols()), lhsSize(xpr.lhs().size()),
                               rhsRows(xpr.rhs().rows()), rhsCols(xpr.rhs().cols()), rhsSize(xpr.rhs().size()) {}
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const BinaryOp& func() const { return op; }
    BinaryOp op;
    evaluator<Lhs> lhsImpl;
    evaluator<Rhs> rhsImpl;
    const internal::variable_if_dynamic<Index, XprType::LhsRowsAtCompileTime> lhsRows;
    const internal::variable_if_dynamic<Index, XprType::LhsColsAtCompileTime> lhsCols;
    const internal::variable_if_dynamic<Index, XprType::LhsSizeAtCompileTime> lhsSize;
    const internal::variable_if_dynamic<Index, XprType::RhsRowsAtCompileTime> rhsRows;
    const internal::variable_if_dynamic<Index, XprType::RhsColsAtCompileTime> rhsCols;
    const internal::variable_if_dynamic<Index, XprType::RhsSizeAtCompileTime> rhsSize;
  };

  Data m_d;
};

template<typename BinaryOp, typename Lhs, typename Rhs>
inline CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs> MakeCoherentCwiseBinaryOp(const Lhs& lhs, const Rhs &rhs) {
  return CoherentCwiseBinaryOp<BinaryOp, Lhs, Rhs>(lhs, rhs);
}

} // namespace internal
} // end namespace Eigen

#endif // EIGEN_CWISE_BINARY_OP_H
