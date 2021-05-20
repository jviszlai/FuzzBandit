#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;

/***
 * ActionIds:
 * 0  - 
 * 1  - 
 * 2  - 
 * 3  -
 * 4  -
 * 5  -
 * 6  -
 * 7  -
 * 8  -
 * 9  -
 * 10 -
 * 11 -
 * 12 -
 * 13 -
 * 14 -
 * 15 -
 ***/
#define NUM_ACTIONS 1166

/***
 * DomainIds:
 * 0 - Mem
 * 1 - Cmp
 * 2 - 
 * 3 - 
 ***/
#define NUM_DOMAINS 1

// The minimum probability to play
// TODO: make this a parameter into gen_input?
// TODO: actually set the right parameter for GAMMA
#define P_MIN 0.01
#define GAMMA 0.01 

std::map<unsigned int, double> weight_map;
std::set<unsigned int> visited_hashes;


// C API
extern "C" {
    int gen_input(char crash_reports[], int domain_reports[NUM_ACTIONS][NUM_DOMAINS], unsigned int curr_hash, unsigned int mutated_hash[NUM_ACTIONS]);
}

int get_advice(int domain_reports[NUM_ACTIONS][NUM_DOMAINS], std::vector<double> &advice);
int get_mix_prob(const double p_min, std::vector<double> &prob);
int sample_mutation(const double total_prob, const std::vector<double> &prob);


// Everytime this is called, do one iteration of multi-armed bandits
// Write file into queue location
// Return 0 if successs, or 1 otherwise
int gen_input(char crash_reports[], int domain_reports[NUM_ACTIONS][NUM_DOMAINS], unsigned int curr_hash, unsigned int mutated_hash[NUM_ACTIONS]) {
    // Compute advice
    std::vector<double> advice(NUM_ACTIONS);
    get_advice(domain_reports, advice);
    
    // Compute unmixed probabilities
    std::vector<double> prob(NUM_ACTIONS);
    double total_prob = 0;
    for (int i = 0; i < prob.size(); i++) { 
        if (weight_map.find(mutated_hash[i]) == weight_map.end()) {
            weight_map[mutated_hash[i]] = 1;
        }
        prob[i] = weight_map[mutated_hash[i]] * advice[i];
        total_prob = total_prob + prob[i];
    }
    for (int i = 0; i < prob.size(); i++) {
        prob[i] = prob[i] / total_prob;
    }

    // Ensure that all actions are played with p_min
    get_mix_prob(P_MIN, prob);

    // Sample action
    int mutation = sample_mutation(total_prob, prob);

    // Compute reward estimator
    std::vector<double> reward_est(NUM_ACTIONS);
    for (int i = 0; i < reward_est.size(); i++) {
        reward_est[i] = crash_reports[i] / prob[i];
    }

    // Compute an exponential weight update
    for (int i = 0; i < NUM_ACTIONS; i++) {
        weight_map[mutated_hash[i]] = weight_map[mutated_hash[i]] * exp((P_MIN / 2) * advice[i] * (reward_est[i] + GAMMA / prob[i]));
    }

    // Add x_t to the seen set
    visited_hashes.insert(curr_hash);

    // Output sampled mutation
    
    return mutation;
}

// Returns the aggregate score from the domain_reports
// Create file, call add_to_queue(filename, length, something else that's 0) 
int get_advice(int domain_reports[NUM_ACTIONS][NUM_DOMAINS], std::vector<double> &advice) {
    // TODO: implement me
    return 0;
}

// Algorithm 2: mixes p_min into prob to improve sampling performance.
int get_mix_prob(const double p_min, std::vector<double> &prob) {
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
            delta += + prob[i] - p_old;
            s -= p_old;
        }
    }
}

// Returns the index of a randomly sampled element
int sample_mutation(const double total_prob, const std::vector<double> &prob) {
    // Compute the cumulant
    std::vector<double> cumulant(prob.size());
    double curr_prob = 0;
    for (int i = 0; i < prob.size(); i++) {
        cumulant[i] = prob[i] + curr_prob;
        curr_prob += prob[i];
    }

    // Sample an index
    double sample = rand() / RAND_MAX;
    int i;
    for (i = 0; i < cumulant.size(); i++) {
        if (sample < cumulant[i]) {
            return i;
        }
    }
    return i;
}
