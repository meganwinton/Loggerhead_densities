
#include <TMB.hpp>
// Function for detecting NAs
template<class Type>
bool isNA(Type x){
  return R_IsNA(asDouble(x));
}

// Space time
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_INTEGER( n_s );
  DATA_INTEGER( n_t );

  DATA_VECTOR( a_s );  // Area associated with location s
  DATA_VECTOR( c_i );  // counts for observation i
  DATA_FACTOR( s_i );  // Random effect index for observation i
  DATA_FACTOR( t_i );  // Random effect index for observation i

  // SPDE objects
  DATA_SPARSE_MATRIX(M0);
  DATA_SPARSE_MATRIX(M1);
  DATA_SPARSE_MATRIX(M2);

  // Parameters
  PARAMETER( beta0 );
  PARAMETER( ln_tau_O ); //variance of spatial part
  PARAMETER( ln_tau_E ); //variance of spatiotemporal part
  PARAMETER( ln_kappa ); //decorrelation distance - assume the same for both

  // Random effects
  PARAMETER_VECTOR( omega_s );   //space
  PARAMETER_ARRAY( epsilon_st ); //space-time

  // Objective funcction
  using namespace density;
  int n_i = c_i.size();
  vector<Type> jnll_comp(3); //keep track of contribution of fixed and both random effects
  jnll_comp.setZero();

  // Derived quantities - writing out eqn for range and sigma from notes - note that v param here is 1
  Type Range = sqrt(8) / exp( ln_kappa );
  Type SigmaO = 1 / sqrt(4 * M_PI * exp(2*ln_tau_O) * exp(2*ln_kappa));
  Type SigmaE = 1 / sqrt(4 * M_PI * exp(2*ln_tau_E) * exp(2*ln_kappa));

  // Probability of random effects 
  //Q is combo of 3 matrices - eqn from last week
  //SCALE function to 'add-in' variance since not directly included
  //the way Q is defined is why we need to divide by the SD (tau)
    //transform density that you want to estimate variance (which starts at 1)
    //doing integral across scaled/transformed random effect
    //SCALE does Jacobian matrix comp for you, which needs when integrating over transformed
  //GMRF gives density of MVN for a sparse matrix
  Eigen::SparseMatrix<Type> Q = exp(4*ln_kappa)*M0 + Type(2.0)*exp(2*ln_kappa)*M1 + M2;
  jnll_comp(1) += SCALE( GMRF(Q), 1/exp(ln_tau_O) )( omega_s ); //space
  for( int t=0; t<n_t; t++){
    jnll_comp(2) += SCALE( GMRF(Q), 1/exp(ln_tau_E) )( epsilon_st.col(t) ); //prob over each time step
  }

  // Log density and abundance at each site - combo of fixed and random effect at each site
  //Note that space is arranged by row and time by column
  array<Type> log_d_st( n_s, n_t );
  for( int t=0; t<n_t; t++){
  for( int s=0; s<n_s; s++){
    log_d_st(s,t) = beta0 + omega_s(s) + epsilon_st(s,t);
  }}

  // Probability of data conditional on random effects - dpois(observed counts at site i, mean at each site in each time step as predicted above)
  for( int i=0; i<n_i; i++){
    if( !isNA(c_i(i)) ) jnll_comp(0) -= dpois( c_i(i), exp(log_d_st(s_i(i),t_i(i))), true );
    //if(c_i(i)==Type(0)) jnll_comp(0) -= log(zero_prob + (Type(1)-zero_prob)*dpois( c_i(i), exp(log_d_st(s_i(i),t_i(i))), false));
    //if(c_i(i)!=Type(0)) jnll_comp(0) -= log(Type(1)-zero_prob) + dpois( c_i(i), exp(beta0 + exp(log_d_st(s_i(i),t_i(i))), true );
  }

  // Reporting
  Type jnll = jnll_comp.sum();
  REPORT( jnll_comp );
  REPORT( jnll );
  ADREPORT( Range );
  ADREPORT( SigmaE );
  ADREPORT( SigmaO );
  REPORT( log_d_st );

  return jnll;
}
