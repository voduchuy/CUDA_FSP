#include "KryExpvFSP.h"

namespace cme {
void KryExpvFSP::step()
{
        double beta, s, avnorm, xm, err_loc;

        beta = norm(sol_vec, 2);

        if (i_step == 0)
        {
                xm = 1.0/double(m);
                //double anorm { norm(A, 1) };
                double fact = pow( (m+1)/exp(1.0), m+1 )*sqrt( 2*(3.1416)*(m+1) );
                t_new = (1.0/anorm)*pow( (fact*tol)/(4.0*beta*anorm), xm);
                btol = anorm*tol; // tolerance for happy breakdown
        }

        size_t mx;
        size_t mb {m};
        size_t k1 {2};

        double tau = std::min( t_final - t_now, t_new );
        H = arma::zeros( m+2, m+2 );

        V[0] = sol_vec;
        V[0] = (1.0/beta)*V[0];

        size_t istart = 0;
        /* Arnoldi loop */
        for ( size_t j{0}; j < m; j++ )
        {
                V[j+1] = matvec(V[j]);

                /* Orthogonalization */
                if (IOP) istart = (j - q_iop + 1 >= 0 ) ? j - q_iop + 1 : 0;

                for ( size_t i { istart }; i <= j; i++ )
                {
                        H(i,j) = arma::dot(V[j+1], V[i]);
                        V[j+1] = V[j+1] - H(i,j)*V[i];
                }
                s = norm(V[j+1], 2);

                if ( s< btol )
                {
                        k1 = 0;
                        mb = j+1;
                        tau = t_final - t_now;
#ifdef KEXPV_VERBOSE
                        std::cout << "happy breakdown.\n";
#endif
                        break;
                }

                H( j+1, j ) = s;
                V[j+1] = V[j+1]/s;
        }


        if ( k1 != 0 )
        {
                H( m+1, m ) = 1.0;
                av = matvec(V[mb]);
        }

        size_t ireject {0};
        while ( ireject < max_reject )
        {
                mx = mb + k1;
                F = expmat(tau*H);
                //std::cout << F << std::endl;
                if ( k1 == 0 )
                {
                        err_loc = btol;
                        break;
                }
                else
                {
                        double phi1 = std::abs( beta*F( m, 0) );
                        double phi2 = std::abs( beta*F( m+1, 0)*avnorm );

                        if ( phi1 > phi2*10.0 )
                        {
                                err_loc = phi2;
                                xm = 1.0/double(m);
                        }
                        else if ( phi1 > phi2 )
                        {
                                err_loc = (phi1*phi2)/(phi1-phi2);
                                xm = 1.0/double(m);
                        }
                        else
                        {
                                err_loc = phi1;
                                xm = 1.0/double(m-1);
                        }
                }

                if ( err_loc <= delta*tau*tol )
                {
                        break;
                }
                else
                {
                        tau = gamma * tau * pow( tau*tol/err_loc, xm );
                        double s = pow( 10.0, floor(log10(tau)) - 1 );
                        tau = ceil(tau/s) * s;
                        if (ireject == max_reject)
                        {
                                // This part could be dangerous, what if one processor exits but the others continue
                                std::cout <<  "Maximum number of failed steps reached.";
                                t_now = t_final;
                                break;
                        }
                        ireject++;
                }
        }

        mx = mb + (size_t) std::max( 0, (int) k1-1 );
        arma::Col<double> F0(mx);
        for (size_t ii{0}; ii < mx; ++ii)
        {
                F0(ii) = beta*F(ii, 0);
        }


        sol_vec *= F0(0)/beta;
        for ( size_t i{1}; i < mx; i++)
        {
                sol_vec += F0(i)*V[i];
        }

        t_now = t_now + tau;
        t_new = gamma*tau*pow( tau*tol/err_loc, xm );
        s = pow( 10.0, floor(log10(t_new) ) - 1.0);
        t_new = ceil( t_new/s )*s;

#ifdef KEXPV_VERBOSE
        std:cout << "t_now = " << t_now << "\n";
#endif
        i_step++;

}

void KryExpvFSP::solve()
{
        while (!final_time_reached())
        {
                step();
        }
}
}
