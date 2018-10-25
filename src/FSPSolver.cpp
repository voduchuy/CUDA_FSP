//
// Created by Huy Vo on 8/10/18.
//

#include "FSPSolver.h"

namespace cme
{
    void FSPSolver::next_time()
    {
        assert( current_time_index < times.n_elem);
        expv.update_final_time(times(current_time_index));
        expv.solve();
        current_time_index+=1;
    }

    void FSPSolver::solve()
    {
        expv.update_final_time(max(times));
        expv.solve();
        current_time_index = times.n_elem-1;
    }
}