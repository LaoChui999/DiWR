#ifndef IGL_FAST_WINDING_NUMBER
#define IGL_FAST_WINDING_NUMBER
#include "igl_inline.h"
#include <Eigen/Core>
#include <vector>
namespace igl
{
  // Generate the precomputation for the fast winding number for point data
  // [Barill et. al 2018].
  //
  // Given a set of 3D points P, with normals N, areas A, along with octree
  // data, and an expansion order, we define a taylor series expansion at each
  // octree cell.
  //
  // The octree data is designed to come from igl::octree, and the areas (if not
  // obtained at scan time), may be calculated using
  // igl::copyleft::cgal::point_areas.
  //
  // Inputs:
  //   P  #P by 3 list of point locations
  //   N  #P by 3 list of point normals
  //   A  #P by 1 list of point areas
  //   point_indices  a vector of vectors, where the ith entry is a vector of
  //                  the indices into P that are the ith octree cell's points
  //   CH             #OctreeCells by 8, where the ith row is the indices of
  //                  the ith octree cell's children
  //   expansion_order    the order of the taylor expansion. We support 0,1,2.
  // Outputs:
  //   CM  #OctreeCells by 3 list of each cell's center of mass
  //   R   #OctreeCells by 1 list of each cell's maximum distance of any point
  //       to the center of mass
  //   EC  #OctreeCells by #TaylorCoefficients list of expansion coefficients.
  //       (Note that #TaylorCoefficients = ∑_{i=1}^{expansion_order} 3^i)
  //
  // See also: igl::copyleft::cgal::point_areas, igl::knn
  template <
    typename DerivedP, 
    typename DerivedA, 
    typename DerivedN,
    typename Index, 
    typename DerivedCH, 
    typename DerivedCM, 
    typename DerivedR,
    typename DerivedEC>
  IGL_INLINE void fast_winding_number(
    const Eigen::MatrixBase<DerivedP>& P,
    const Eigen::MatrixBase<DerivedN>& N,
    const Eigen::MatrixBase<DerivedA>& A,
    const std::vector<std::vector<Index> > & point_indices,
    const Eigen::MatrixBase<DerivedCH>& CH,
    const int expansion_order,
    Eigen::PlainObjectBase<DerivedCM>& CM,
    Eigen::PlainObjectBase<DerivedR>& R,
    Eigen::PlainObjectBase<DerivedEC>& EC);
}
#ifndef IGL_STATIC_LIBRARY
#  include "fast_winding_number.cpp"
#endif

#endif
