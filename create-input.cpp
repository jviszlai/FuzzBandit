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

/***
 * DomainIds:
 * 0 - Mem
 * 1 - Cmp
 * 2 - 
 * 3 - 
 ***/
#define NUM_DOMAINS 2

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
double domain_weights[] = {1, 10000000000000};

// C API
extern "C"
{
    typedef struct {
        char *out_buf;
        int len;
        char fault;
        char **argv;
        char *op_descript;
    } mutation_buf;

    typedef struct mutation {
        int stage_id;
        int index;
        int feedback[NUM_DOMAINS];
        unsigned int hash;
        int did_crash;
        mutation_buf file;
        struct mutation *next;
    } mutation;

    int gen_input(mutation *mutation_list, unsigned int curr_hash);
}

// Function declarations
int get_advice(mutation *mutation_list, std::vector<double> &advice);
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
int gen_input(mutation *mutation_list, unsigned int curr_hash)
{
    log_file.open(log_file_name, std::ios_base::app);


    log_file << "================================\n\n\n REPORTS \n\n\n\ ";
    mutation *curr = mutation_list;
    while (curr) {
        log_file << curr->feedback[1] << ", ";
        curr = curr->next;
    }

    // Compute advice
    std::vector<double> advice(0);
    get_advice(mutation_list, advice);

    // Compute unmixed probabilities
    std::vector<double> prob(0);
    double total_prob = 0;
    
    curr = mutation_list;
    for (int i = 0; curr; curr = curr->next, i++)
    {
        if (weight_map.find(curr->hash) == weight_map.end())
        {
            weight_map[curr->hash] = 1;
        }
        prob.emplace_back(weight_map[curr->hash] * advice[i]);
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
    std::vector<double> reward_est(0);
    curr = mutation_list;
    for (int i = 0; curr; curr = curr->next, i++)
    {
        reward_est.emplace_back(curr->did_crash / prob[i]);
    }

    // Compute an exponential weight update
    curr = mutation_list;
    for (int i = 0; curr; curr = curr->next, i++)
    {
        weight_map[curr->hash] = weight_map[curr->hash] * exp((P_MIN / 2) * advice[i] * (reward_est[i] + GAMMA / prob[i]));
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
int get_advice(mutation *mutation_list, std::vector<double> &advice)
{
    // Compute aggregate score for each action
    double total_score = 0;
    for (mutation *curr = mutation_list; curr; curr = curr->next)
    {
        double aggregate_score = 0;
        for (int j = 0; j < NUM_DOMAINS; j++)
        {
            aggregate_score += domain_weights[j] * curr->feedback[j];
        }
        advice.emplace_back(aggregate_score);
        total_score += aggregate_score;
    }

    // Normalize the feedback
    for (int i = 0; i < advice.size(); i++)
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
