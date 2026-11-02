#include "fast_winding_number.h"
#include <cassert>
#include <functional>

template <
  typename DerivedP, 
  typename DerivedA, 
  typename DerivedN,
  typename Index, 
  typename DerivedCH, 
  typename DerivedCM, 
  typename DerivedR,
  typename DerivedEC>
IGL_INLINE void igl::fast_winding_number(
  const Eigen::MatrixBase<DerivedP>& P,
  const Eigen::MatrixBase<DerivedN>& N,
  const Eigen::MatrixBase<DerivedA>& A,
  const std::vector<std::vector<Index> > & point_indices,
  const Eigen::MatrixBase<DerivedCH>& CH,
  const int expansion_order,
  Eigen::PlainObjectBase<DerivedCM>& CM,
  Eigen::PlainObjectBase<DerivedR>& R,
  Eigen::PlainObjectBase<DerivedEC>& EC)
{
  typedef typename DerivedP::Scalar real_p;
  typedef typename DerivedN::Scalar real_n;
  typedef typename DerivedA::Scalar real_a;
  typedef typename DerivedCM::Scalar real_cm;
  typedef typename DerivedR::Scalar real_r;
  typedef typename DerivedEC::Scalar real_ec;

  typedef Eigen::Matrix<real_p,1,3> RowVec3p;

  int m = CH.size();
  int num_terms;

  assert(expansion_order < 3 && expansion_order >= 0 && "m must be less than n");
  if(expansion_order == 0){
      num_terms = 3;
  } else if(expansion_order ==1){
      num_terms = 3 + 9;
  } else if(expansion_order == 2){
      num_terms = 3 + 9 + 27;
  }

  R.resize(m);
  CM.resize(m,3);
  EC.resize(m,num_terms);
  EC.setZero(m,num_terms);
  std::function< void(const int) > helper;
  helper = [&helper,
            &P,&N,&A,&expansion_order,&point_indices,&CH,&EC,&R,&CM]
  (const int index)-> void
  {
      Eigen::Matrix<real_cm,1,3> masscenter;
      masscenter << 0,0,0;
      Eigen::Matrix<real_ec,1,3> zeroth_expansion;
      zeroth_expansion << 0,0,0;
      real_p areatotal = 0.0;
      for(int j = 0; j < point_indices[index].size(); j++){
          int curr_point_index = point_indices[index][j];
        
          areatotal += A(curr_point_index);
          masscenter += A(curr_point_index)*P.row(curr_point_index);
          zeroth_expansion += A(curr_point_index)*N.row(curr_point_index);
      }
    
      masscenter = masscenter/areatotal;
      CM.row(index) = masscenter;
      EC.block(index,0,1,3) = zeroth_expansion;
    
      real_r max_norm = 0;
      real_r curr_norm;
    
      for(int i = 0; i < point_indices[index].size(); i++){
          //Get max distance from center of mass:
          int curr_point_index = point_indices[index][i];
          Eigen::Matrix<real_r,1,3> point =
              P.row(curr_point_index)-masscenter;
          curr_norm = point.norm();
          if(curr_norm > max_norm){
              max_norm = curr_norm;
          }
        
          //Calculate higher order terms if necessary
          Eigen::Matrix<real_ec,3,3> TempCoeffs;
          if(EC.cols() >= (3+9)){
              TempCoeffs = A(curr_point_index)*point.transpose()*
                              N.row(curr_point_index);
              EC.block(index,3,1,9) +=
              Eigen::Map<Eigen::Matrix<real_ec,1,9> >(TempCoeffs.data(),
                                                      TempCoeffs.size());
          }
        
          if(EC.cols() == (3+9+27)){
              for(int k = 0; k < 3; k++){
                  TempCoeffs = 0.5 * point(k) * (A(curr_point_index)*
                                point.transpose()*N.row(curr_point_index));
                  EC.block(index,12+9*k,1,9) += Eigen::Map<
                    Eigen::Matrix<real_ec,1,9> >(TempCoeffs.data(),
                                                 TempCoeffs.size());
              }
          }
      }
    
      R(index) = max_norm;
      if(CH(index,0) != -1)
      {
          for(int i = 0; i < 8; i++){
              int child = CH(index,i);
              helper(child);
          }
      }
  };
  helper(0);
}
