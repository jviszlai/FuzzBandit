#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>

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
#define NUM_ACTIONS 1137

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
#define P_MIN 0.0001
#define GAMMA 0.0005

std::map<unsigned int, double> weight_map;
std::set<unsigned int> visited_hashes;

static ofstream log_file;
static std::string log_file_name = "prob_log.txt";

// Global array of weights on the domain
double domain_weights[] = {1};

// C API
extern "C"
{
    int gen_input(char crash_reports[], int (*domain_reports)[NUM_ACTIONS][NUM_DOMAINS], unsigned int curr_hash, unsigned int mutated_hash[NUM_ACTIONS]);
}

// Function declarations
int get_advice(int (*domain_reports)[NUM_ACTIONS][NUM_DOMAINS], std::vector<double> &advice);
int get_mix_prob(const double p_min, std::vector<double> &prob);
int sample_mutation(const double total_prob, const std::vector<double> &prob);

// Comparator used to iterate through probabilities by order of magnitude
struct IndexComparator
{
    explicit IndexComparator(const std::vector<double> &values) : values(&values) {}

    bool operator()(size_t a, size_t b)
    {
        return (*values)[a] < (*values)[b];
    }

private:
    const std::vector<double> *values;
};

// Everytime this is called, do one iteration of multi-armed bandits
// Write file into queue location
// Return 0 if successs, or 1 otherwise
int gen_input(char crash_reports[], int (*domain_reports)[NUM_ACTIONS][NUM_DOMAINS], unsigned int curr_hash, unsigned int mutated_hash[NUM_ACTIONS])
{
    log_file.open(log_file_name, std::ios_base::app);


    log_file << "================================\n\n\n REPORTS \n\n\n\ ";
    for (int i = 0; i < 100; i++) {
        log_file << (*domain_reports)[i][1] << ", ";
    }

    // Compute advice
    std::vector<double> advice(NUM_ACTIONS);
    get_advice(domain_reports, advice);

    // Compute unmixed probabilities
    std::vector<double> prob(NUM_ACTIONS);
    double total_prob = 0;
    for (int i = 0; i < prob.size(); i++)
    {
        if (weight_map.find(mutated_hash[i]) == weight_map.end())
        {
            weight_map[mutated_hash[i]] = 1;
        }
        prob[i] = weight_map[mutated_hash[i]] * advice[i];
        total_prob = total_prob + prob[i];
    }
    for (int i = 0; i < prob.size(); i++)
    {
        prob[i] = prob[i] / total_prob;
    }

    // Ensure that all actions are played with p_min
    get_mix_prob(P_MIN, prob);

    // Sample action
    int mutation = sample_mutation(total_prob, prob);

    // Compute reward estimator
    std::vector<double> reward_est(NUM_ACTIONS);
    for (int i = 0; i < reward_est.size(); i++)
    {
        reward_est[i] = crash_reports[i] / prob[i];
    }

    // Compute an exponential weight update
    for (int i = 0; i < NUM_ACTIONS; i++)
    {
        weight_map[mutated_hash[i]] = weight_map[mutated_hash[i]] * exp((P_MIN / 2) * advice[i] * (reward_est[i] + GAMMA / prob[i]));
    }

    // Add x_t to the seen set
    visited_hashes.insert(curr_hash);

    std::ostringstream oss;
    log_file << "\n\n\n PROB \n\n\n\ ";
    std::copy(prob.begin(), prob.end()-1000,
        std::ostream_iterator<double>(oss, ", "));

    log_file << oss.str();

    std::ostringstream oss2;
    log_file << "\n\n\n ADVICE \n\n\n\ ";
    std::copy(advice.begin(), advice.end()-1000,
        std::ostream_iterator<double>(oss2, ", "));

    log_file << oss2.str();
    log_file << "\n\n MUTATION \n\n";
    log_file << mutation;


    log_file.close();
    // Output sampled mutation
    return mutation;
}

// Returns the aggregate score from the domain_reports
int get_advice(int (*domain_reports)[NUM_ACTIONS][NUM_DOMAINS], std::vector<double> &advice)
{
    // Compute aggregate score for each action
    double total_score = 0;
    for (int i = 0; i < NUM_ACTIONS; i++)
    {
        double aggregate_score = 0;
        for (int j = 0; j < NUM_DOMAINS; j++)
        {
            aggregate_score += domain_weights[j] * (*domain_reports)[i][j];
            if ((*domain_reports)[i][j] < 0) {
                log_file << "\n\nBIG MEME WHY :(\n\n" << (*domain_reports)[i][j];
            }
        }
        if (aggregate_score < 0) {
            log_file << "\n\nAGGREGATE BIG MEME WHY :(\n\n" << aggregate_score;
        }
        advice[i] = aggregate_score;
        total_score += aggregate_score;
    }

    // Normalize the feedback
    for (int i = 0; i < NUM_ACTIONS; i++)
    {
        advice[i] = advice[i] / total_score;
    }
}

// Algorithm 2: mixes p_min into prob to improve sampling performance.
int get_mix_prob(const double p_min, std::vector<double> &prob)
{
    // Rescaling variables
    double delta = 0;
    double s = 1;

    // Get indices of prob sorted by magnitude of prob
    std::vector<size_t> indices(prob.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), IndexComparator(prob));

    // Iterate through the probabilities in sorted order
    for (const auto i : indices)
    {
        double p_old = prob[i];

        // Set the new probabilities
        if (p_old * (1 - delta / s) >= p_min)
        {
            prob[i] = p_old * (1 - delta / s);
        }
        else
        {
            prob[i] = p_min;
            delta += +prob[i] - p_old;
            s -= p_old;
        }
    }
}

// Returns the index of a randomly sampled element
int sample_mutation(const double total_prob, const std::vector<double> &prob)
{
    // Compute the cumulant
    std::vector<double> cumulant(prob.size());
    double curr_prob = 0;
    for (int i = 0; i < prob.size(); i++)
    {
        cumulant[i] = prob[i] + curr_prob;
        curr_prob += prob[i];
    }

    // Sample an index
    double sample = rand() / (1.0 * RAND_MAX);
    int i;
    for (i = 0; i < cumulant.size(); i++)
    {
        if (sample < cumulant[i])
        {
            return i;
        }
    }
    return i;
}
