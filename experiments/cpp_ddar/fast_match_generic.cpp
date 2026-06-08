// Experimental C++ accelerator for AlphaGeometry DD generic theorem matching.
//
// This deliberately implements only the pure backtracking core used by
// dd.try_to_map / dd.match_generic. Graph mutation, dependency construction,
// algebraic reasoning, and numerical checks remain in Python for this first
// equivalence/speed experiment.

#include <algorithm>
#include <functional>
#include <vector>

extern "C" int fast_match_generic(
    int num_clauses,
    const int* clause_arities,
    const int* clause_var_offsets,
    const int* clause_vars,
    const int* enum_counts,
    const int* enum_offsets,
    const int* enum_points,
    int num_vars,
    int num_points,
    int max_results,
    int* out_var_points) {
  if (num_clauses < 0 || num_vars < 0 || num_points < 0 || max_results < 0) {
    return -1;
  }
  if (num_clauses == 0 || num_vars == 0 || max_results == 0) {
    return 0;
  }

  std::vector<int> var_to_point(num_vars, -1);
  std::vector<int> point_to_var(num_points, -1);
  int result_count = 0;

  std::function<void(int)> backtrack = [&](int clause_index) {
    if (result_count >= max_results) {
      return;
    }
    if (clause_index == num_clauses) {
      int* out = out_var_points + result_count * num_vars;
      std::copy(var_to_point.begin(), var_to_point.end(), out);
      ++result_count;
      return;
    }

    const int arity = clause_arities[clause_index];
    const int var_offset = clause_var_offsets[clause_index];
    const int enum_offset = enum_offsets[clause_index];
    const int enum_count = enum_counts[clause_index];

    for (int row = 0; row < enum_count && result_count < max_results; ++row) {
      bool fail = false;
      std::vector<int> changed_vars;
      std::vector<int> changed_points;

      for (int pos = 0; pos < arity; ++pos) {
        const int var_id = clause_vars[var_offset + pos];
        const int point_id = enum_points[enum_offset + row * arity + pos];
        if (var_id < 0 || var_id >= num_vars || point_id < 0 ||
            point_id >= num_points) {
          fail = true;
          break;
        }

        const int old_point = var_to_point[var_id];
        if (old_point != -1 && old_point != point_id) {
          fail = true;
          break;
        }
        const int old_var = point_to_var[point_id];
        if (old_var != -1 && old_var != var_id) {
          fail = true;
          break;
        }

        if (old_point == -1) {
          var_to_point[var_id] = point_id;
          changed_vars.push_back(var_id);
        }
        if (old_var == -1) {
          point_to_var[point_id] = var_id;
          changed_points.push_back(point_id);
        }
      }

      if (!fail) {
        backtrack(clause_index + 1);
      }

      for (int var_id : changed_vars) {
        var_to_point[var_id] = -1;
      }
      for (int point_id : changed_points) {
        point_to_var[point_id] = -1;
      }
    }
  };

  backtrack(0);
  return result_count;
}
