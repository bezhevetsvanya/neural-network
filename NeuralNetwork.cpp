#include <iostream>
#include <vector>
#include <random>

class NeuralNetwork {
public:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    int num_layers;

    NeuralNetwork(const std::vector<int>& layer_sizes) {
        num_layers = layer_sizes.size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0, 1);

        for (int i = 0; i < num_layers - 1; i++) {
            std::vector<double> layer_weights;
            for (int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
                layer_weights.push_back(dis(gen));
            }
            weights.push_back(layer_weights);
            biases.push_back(dis(gen));
        }
    }

    std::vector<double> feedforward(const std::vector<double>& input) {
        std::vector<double> output = input;
        for (int i = 0; i < num_layers - 1; i++) {
            std::vector<double> next_output;
            for (int j = 0; j < weights[i].size() / output.size(); j++) {
                double sum = 0;
                for (int k = 0; k < output.size(); k++) {
                    sum += output[k] * weights[i][j * output.size() + k];
                }
                sum += biases[i];
                next_output.push_back(sigmoid(sum));
            }
            output = next_output;
        }
        return output;
    }

private:
    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }
};

int main() {
    NeuralNetwork network({2, 2, 1});
    std::vector<double> input = {1, 0};
    std::vector<double> output = network.feedforward(input);
    std::cout << "Output: " << output[0] << std::endl;
    return 0;
}
