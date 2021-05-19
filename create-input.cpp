#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;
/***
 * ActionIds:
 * 0 - 
 * 1 - 
 * 2 - 
 * 3 -
 * ...
 * 
 ***/
#define NUM_ACTIONS 16

/***
 * DomainIds:
 * 0 - Mem
 * 1 - Cmp
 * 
 ***/
#define NUM_DOMAINS 4


// Need x_t, and all mutations on x_t

// Everytime this is called, do one iteration of multi-armed bandits
// Write file into queue location
// Return 0 if successs, or 1 otherwise
int gen_input(char crash_reports[], int domain_reports[NUM_ACTIONS][NUM_DOMAINS]) {
    // Compute advice
    // Compute unmixed probabilities
    // Call mix_prob
    // Sample action
    // Compute reward vector
    // Compute reward estimator
    // Compute weight update
    // Add x_t to seen set
    // Update with new input
}

// Returns the aggregate score from the domain_reports
// Create file, call add_to_queue(filename, length, something else that's 0) 
int get_advice(int domain_reports[NUM_ACTIONS][NUM_DOMAINS], std::vector<double> &advice) {
    // TODO: implement me
    return 0;
}

// Algorithm 2 in the paper
int mix_prob(const double p_min, std::vector<double> &prob) {
    // Rescaling variables
    double delta = 0;
    double s = 0;

    // Get indices of prob sorted by magnitude of prob
    std::vector<size_t> indices(prob.size());
    iota(indices.begin(), indices.end(), 0);
    stable_sort(indices.begin(), indices.end(), 
        [&prob](size_t a, size_t b){return prob[a] < prob[b];});

    // Iterate through the probabilities in sorted order
    for (const auto i: indices) {
        double p_old = prob[i];

        // Set the new probabilities
        if (p_old * (1 - delta / s) >= p_min) {
            prob[i] = p_old * (1 - delta/s);
        } else {
            prob[i] = p_min;
            delta = delta + prob[i] - p_old;
            s = s - p_old;
        }
    }
}

